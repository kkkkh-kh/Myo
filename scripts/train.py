import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


ROOT_DIR = Path(os.environ.get("ROOT_DIR", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / "datasets"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", ROOT_DIR / "configs" / "default.yaml"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT_DIR / "checkpoints"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

from data.dataset import GlossChineseDataset
from data.gloss_noise_augment import GlossNoiseAugmentor
from data.preprocess import extract_corpora, read_parallel_pairs
from data.vocabulary import Vocabulary
from model.decoder import ChineseDecoder
from model.encoder import GlossEncoder
from model.seq2seq import Seq2Seq
from train.checkpointing import load_checkpoint_into_model
from train.trainer import Trainer


def resolve_dataset_path(data_dir: Path, *candidates: str) -> Path:
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing dataset file, tried: {', '.join(str(data_dir / c) for c in candidates)}")


def resolve_train_path(data_dir: Path, config: dict) -> Path:
    train_file_override = os.environ.get("TRAIN_FILE")
    if train_file_override:
        return resolve_dataset_path(data_dir, train_file_override)

    augment_cfg = config.get("word_order_augment", {})
    if augment_cfg.get("enabled", False):
        for candidate in ("train_augmented.tsv", "train_augmented.csv"):
            path = data_dir / candidate
            if path.exists():
                return path

    return resolve_dataset_path(data_dir, "train.tsv", "train.csv")


def _candidate_paths(raw_path: str, *roots: Path) -> list[Path]:
    base = Path(raw_path)
    candidates: list[Path] = [base]
    for root in roots:
        if not base.is_absolute():
            candidates.append(root / base)
    if base.parts and base.parts[0].lower() == "myo" and len(base.parts) > 1:
        stripped = Path(*base.parts[1:])
        for root in roots:
            candidates.append(root / stripped)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_resume_path(root_dir: Path, output_dir: Path, config: dict) -> Optional[Path]:
    configured = config.get("train", {}).get("resume_from")
    resume_value = os.environ.get("RESUME_FROM") or configured
    if resume_value:
        for candidate in _candidate_paths(str(resume_value), output_dir, root_dir, Path.cwd()):
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_value}")

    if _env_flag("AUTO_RESUME", default=False):
        auto_path = output_dir / "latest_model.pt"
        if auto_path.exists():
            return auto_path.resolve()
    return None


def build_vocabularies(train_path: Path, config: dict) -> tuple[Vocabulary, Vocabulary]:
    train_pairs = read_parallel_pairs(train_path.as_posix())
    zh_tokenizer_mode = config.get("data", {}).get("zh_tokenizer", "char")
    gloss_corpus, zh_corpus = extract_corpora(train_pairs, zh_tokenizer_mode=zh_tokenizer_mode)

    gloss_vocab = Vocabulary()
    gloss_vocab.build_from_corpus(gloss_corpus, max_size=config["model"]["gloss_vocab_size"])
    zh_vocab = Vocabulary()
    zh_vocab.build_from_corpus(zh_corpus, max_size=config["model"]["zh_vocab_size"])
    return gloss_vocab, zh_vocab


def load_or_build_vocabularies(
    train_path: Path,
    config: dict,
    checkpoint_dir: Optional[Path] = None,
    reuse_checkpoint_vocab: bool = False,
) -> tuple[Vocabulary, Vocabulary, str]:
    if reuse_checkpoint_vocab and checkpoint_dir is not None:
        gloss_path = checkpoint_dir / "gloss_vocab.json"
        zh_path = checkpoint_dir / "zh_vocab.json"
        if gloss_path.exists() and zh_path.exists():
            return Vocabulary.load(gloss_path), Vocabulary.load(zh_path), "checkpoint"

    gloss_vocab, zh_vocab = build_vocabularies(train_path, config)
    return gloss_vocab, zh_vocab, "train_data"


def _checkpoint_vocab_sizes(checkpoint: dict) -> Tuple[Optional[int], Optional[int]]:
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    gloss_size: Optional[int] = None
    zh_size: Optional[int] = None

    if isinstance(state_dict, dict):
        gloss_weight = state_dict.get("encoder.embedding.weight")
        zh_weight = state_dict.get("decoder.embedding.weight")
        if torch.is_tensor(gloss_weight):
            gloss_size = int(gloss_weight.size(0))
        if torch.is_tensor(zh_weight):
            zh_size = int(zh_weight.size(0))

    if gloss_size is None or zh_size is None:
        cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        if gloss_size is None and "gloss_vocab_size" in model_cfg:
            gloss_size = int(model_cfg["gloss_vocab_size"])
        if zh_size is None and "zh_vocab_size" in model_cfg:
            zh_size = int(model_cfg["zh_vocab_size"])

    return gloss_size, zh_size


def resume_checkpoint_matches_vocab(
    checkpoint: dict,
    gloss_vocab_size: int,
    zh_vocab_size: int,
) -> tuple[bool, str]:
    ckpt_gloss, ckpt_zh = _checkpoint_vocab_sizes(checkpoint)
    if ckpt_gloss is None or ckpt_zh is None:
        return True, "checkpoint vocab size unavailable"
    if ckpt_gloss != gloss_vocab_size or ckpt_zh != zh_vocab_size:
        return (
            False,
            f"checkpoint vocab mismatch (checkpoint: gloss={ckpt_gloss}, zh={ckpt_zh}; "
            f"current: gloss={gloss_vocab_size}, zh={zh_vocab_size})",
        )
    return True, "checkpoint vocab size matched"


def build_dataloader(dataset: GlossChineseDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=GlossChineseDataset.collate_fn,
    )


def build_noise_augmentor(gloss_vocab: Vocabulary, config: dict) -> Optional[GlossNoiseAugmentor]:
    noise_config = config.get("noise_augment", {})
    if not noise_config.get("enabled", False):
        return None

    candidate_token_ids = [
        token_id
        for token_id in range(len(gloss_vocab))
        if token_id not in {Vocabulary.PAD_ID, Vocabulary.BOS_ID, Vocabulary.EOS_ID}
    ]
    return GlossNoiseAugmentor(
        candidate_token_ids=candidate_token_ids,
        p_del=float(noise_config.get("p_del", 0.05)),
        p_sub=float(noise_config.get("p_sub", 0.05)),
        p_ins=float(noise_config.get("p_ins", 0.03)),
        p_rep=float(noise_config.get("p_rep", 0.03)),
        warmup_ratio=float(noise_config.get("warmup_ratio", 0.2)),
    )


def build_encoder(config: dict) -> GlossEncoder:
    encoder_config = config.get("encoder", {})
    return GlossEncoder(
        gloss_vocab_size=config["model"]["gloss_vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        pad_id=Vocabulary.PAD_ID,
        use_sen=bool(encoder_config.get("use_sen", False)),
        sen_reduction=int(encoder_config.get("sen_reduction", 16)),
        use_transformer=bool(encoder_config.get("use_transformer", False)),
        transformer_layers=int(encoder_config.get("transformer_layers", 2)),
        transformer_heads=int(encoder_config.get("transformer_heads", 4)),
        transformer_dropout=float(encoder_config.get("transformer_dropout", 0.1)),
    )


def main() -> None:
    print(f"ROOT_DIR    : {ROOT_DIR}")
    print(f"DATA_DIR    : {DATA_DIR}")
    print(f"CONFIG_PATH : {CONFIG_PATH}")
    print(f"OUTPUT_DIR  : {OUTPUT_DIR}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config.setdefault("train", {})
    config.setdefault("model", {})
    config.setdefault("data", {})

    explicit_run_id = os.environ.get("RUN_ID")
    default_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_id"] = explicit_run_id or default_run_id
    config["save_dir"] = OUTPUT_DIR.resolve().as_posix()
    config["project_root"] = ROOT_DIR.resolve().as_posix()

    reuse_checkpoint_vocab = _env_flag(
        "REUSE_CHECKPOINT_VOCAB",
        default=bool(config.get("train", {}).get("reuse_checkpoint_vocab", False)),
    )

    resume_path = resolve_resume_path(ROOT_DIR, OUTPUT_DIR, config)
    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path.as_posix(), map_location="cpu")
        print(f"Resume checkpoint: {resume_path}")

    train_path = resolve_train_path(DATA_DIR, config)
    val_path = resolve_dataset_path(DATA_DIR, "val.tsv", "dev.tsv", "val.csv", "dev.csv")
    zh_tokenizer_mode = config.get("data", {}).get("zh_tokenizer", "char")

    resume_vocab_dir = resume_path.parent if resume_path is not None else None
    gloss_vocab, zh_vocab, vocab_source = load_or_build_vocabularies(
        train_path,
        config,
        checkpoint_dir=resume_vocab_dir,
        reuse_checkpoint_vocab=reuse_checkpoint_vocab,
    )

    if resume_checkpoint is not None:
        compatible, reason = resume_checkpoint_matches_vocab(
            resume_checkpoint,
            gloss_vocab_size=len(gloss_vocab),
            zh_vocab_size=len(zh_vocab),
        )
        if not compatible:
            print(f"Resume disabled: {reason}")
            resume_checkpoint = None
            resume_path = None
        elif not explicit_run_id:
            resumed_run_id = str(resume_checkpoint.get("run_id", "")).strip()
            if resumed_run_id:
                config["run_id"] = resumed_run_id

    gloss_vocab.save(OUTPUT_DIR / "gloss_vocab.json")
    zh_vocab.save(OUTPUT_DIR / "zh_vocab.json")

    config["model"]["gloss_vocab_size"] = len(gloss_vocab)
    config["model"]["zh_vocab_size"] = len(zh_vocab)
    with (OUTPUT_DIR / "runtime_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    noise_augmentor = build_noise_augmentor(gloss_vocab, config)
    train_dataset = GlossChineseDataset(
        tsv_path=train_path.as_posix(),
        gloss_vocab=gloss_vocab,
        zh_vocab=zh_vocab,
        max_gloss_len=config["data"]["max_gloss_len"],
        max_zh_len=config["data"]["max_zh_len"],
        zh_tokenizer_mode=zh_tokenizer_mode,
        augment=noise_augmentor is not None,
        augmentor=noise_augmentor,
    )
    val_dataset = GlossChineseDataset(
        tsv_path=val_path.as_posix(),
        gloss_vocab=gloss_vocab,
        zh_vocab=zh_vocab,
        max_gloss_len=config["data"]["max_gloss_len"],
        max_zh_len=config["data"]["max_zh_len"],
        zh_tokenizer_mode=zh_tokenizer_mode,
        augment=False,
        augmentor=None,
    )

    print(f"训练集样本数 : {len(train_dataset)}")
    print(f"验证集样本数 : {len(val_dataset)}")
    print(f"Run ID      : {config['run_id']}")
    print(f"中文切分方式 : {zh_tokenizer_mode}")
    print(f"启用 Gloss 噪声增强 : {noise_augmentor is not None}")
    print(f"训练文件     : {train_path.name}")
    print(f"Vocab source: {vocab_source}")
    print(f"Resume mode : {resume_checkpoint is not None}")

    train_loader = build_dataloader(train_dataset, config["train"]["batch_size"], shuffle=True)
    val_loader = build_dataloader(val_dataset, config["train"]["batch_size"], shuffle=False)

    encoder = build_encoder(config)
    model_cfg = config.get("model", {})
    decoder = ChineseDecoder(
        zh_vocab_size=config["model"]["zh_vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        use_word_order_attention=bool(model_cfg.get("use_word_order_attention", False)),
        max_relative_position=int(model_cfg.get("max_relative_position", 64)),
        use_order_guidance=bool(model_cfg.get("use_order_guidance", True)),
        guidance_lambda_init=float(model_cfg.get("guidance_lambda_init", 1.0)),
        guidance_decay_ratio=float(model_cfg.get("guidance_decay_ratio", 0.3)),
    )

    model = Seq2Seq(encoder=encoder, decoder=decoder)
    optimizer = AdamW(model.parameters(), lr=config["train"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, config=config)

    if resume_checkpoint is not None:
        load_checkpoint_into_model(model, resume_checkpoint, map_location="cpu")
        resumed_epoch = trainer.resume_from_checkpoint(resume_checkpoint)
        print(f"Resume state restored, continue from epoch {resumed_epoch + 1}")

    result = trainer.train(train_loader, val_loader)

    print(
        "训练完成，最佳验证损失：{best_val_loss:.4f}，BLEU-4：{best_bleu4:.2f}，"
        "ROUGE-L：{best_rouge_l:.2f}，WER：{best_wer:.2f}".format(**result)
    )
    print(f"最佳模型路径：{result['best_model_path']}")
    print(f"Latest checkpoint: {result.get('latest_model_path', OUTPUT_DIR / 'latest_model.pt')}")
    print(f"验证样例日志：{result['validation_samples_path']}")


if __name__ == "__main__":
    main()
