import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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
from train.trainer import Trainer


def resolve_dataset_path(data_dir: Path, *candidates: str) -> Path:
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"未找到数据文件，尝试过：{', '.join(str(data_dir / candidate) for candidate in candidates)}"
    )


def build_vocabularies(train_path: Path, config: dict) -> tuple[Vocabulary, Vocabulary]:
    train_pairs = read_parallel_pairs(train_path.as_posix())
    zh_tokenizer_mode = config.get("data", {}).get("zh_tokenizer", "char")
    gloss_corpus, zh_corpus = extract_corpora(train_pairs, zh_tokenizer_mode=zh_tokenizer_mode)

    gloss_vocab = Vocabulary()
    gloss_vocab.build_from_corpus(gloss_corpus, max_size=config["model"]["gloss_vocab_size"])
    zh_vocab = Vocabulary()
    zh_vocab.build_from_corpus(zh_corpus, max_size=config["model"]["zh_vocab_size"])
    return gloss_vocab, zh_vocab


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

    train_path = resolve_dataset_path(DATA_DIR, "train.tsv", "train.csv")
    val_path = resolve_dataset_path(DATA_DIR, "val.tsv", "dev.tsv", "val.csv", "dev.csv")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["run_id"] = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    config["save_dir"] = OUTPUT_DIR.resolve().as_posix()
    config["project_root"] = ROOT_DIR.resolve().as_posix()
    zh_tokenizer_mode = config.get("data", {}).get("zh_tokenizer", "char")

    gloss_vocab, zh_vocab = build_vocabularies(train_path, config)
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

    train_loader = build_dataloader(train_dataset, config["train"]["batch_size"], shuffle=True)
    val_loader = build_dataloader(val_dataset, config["train"]["batch_size"], shuffle=False)

    encoder = build_encoder(config)
    decoder = ChineseDecoder(
        zh_vocab_size=config["model"]["zh_vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )

    model = Seq2Seq(encoder=encoder, decoder=decoder)
    optimizer = AdamW(model.parameters(), lr=config["train"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, config=config)
    result = trainer.train(train_loader, val_loader)

    print(
        "训练完成，最佳验证损失：{best_val_loss:.4f}，BLEU-4：{best_bleu4:.2f}，"
        "ROUGE-L：{best_rouge_l:.2f}，WER：{best_wer:.2f}".format(**result)
    )
    print(f"最佳模型路径：{result['best_model_path']}")
    print(f"验证样例日志：{result['validation_samples_path']}")


if __name__ == "__main__":
    main()
