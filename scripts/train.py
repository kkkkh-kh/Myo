import os
import sys
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import os
from pathlib import Path
import os
from pathlib import Path

# 以当前脚本文件位置为基准，向上找到项目根目录
ROOT_DIR    = Path(os.environ.get("ROOT_DIR",    Path(__file__).resolve().parent))

DATA_DIR    = Path(os.environ.get("DATA_DIR",    ROOT_DIR / "data"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", ROOT_DIR / "configs" / "config.yaml"))
OUTPUT_DIR  = Path(os.environ.get("OUTPUT_DIR",  ROOT_DIR / "checkpoints"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"ROOT_DIR    : {ROOT_DIR}")
print(f"DATA_DIR    : {DATA_DIR}")
print(f"CONFIG_PATH : {CONFIG_PATH}")
print(f"OUTPUT_DIR  : {OUTPUT_DIR}")
import os
from pathlib import Path

# 以当前脚本文件位置为基准，向上找到项目根目录
ROOT_DIR    = Path(os.environ.get("ROOT_DIR",    Path(__file__).resolve().parents[1]))

DATA_DIR    = Path(os.environ.get("DATA_DIR",    ROOT_DIR / "datasets"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", ROOT_DIR / "configs" / "default.yaml"))
OUTPUT_DIR  = Path(os.environ.get("OUTPUT_DIR",  ROOT_DIR / "checkpoints"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

from data.dataset import GlossChineseDataset
from data.preprocess import extract_corpora, read_parallel_pairs
from data.vocabulary import Vocabulary
from model.decoder import ChineseDecoder
from model.encoder import GlossEncoder
from model.seq2seq import Seq2Seq
from train.trainer import Trainer


def resolve_dataset_path(data_dir, *candidates):
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"未找到数据文件，尝试过：{', '.join(str(data_dir / candidate) for candidate in candidates)}"
    )


train_path = resolve_dataset_path(DATA_DIR, "train.tsv", "train.csv")
val_path = resolve_dataset_path(DATA_DIR, "val.tsv", "dev.tsv", "val.csv", "dev.csv")

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

train_pairs = read_parallel_pairs(train_path.as_posix())
val_pairs = read_parallel_pairs(val_path.as_posix())
gloss_corpus, zh_corpus = extract_corpora(train_pairs + val_pairs)

gloss_vocab = Vocabulary()
gloss_vocab.build_from_corpus(gloss_corpus, max_size=config["model"]["gloss_vocab_size"])
zh_vocab = Vocabulary()
zh_vocab.build_from_corpus(zh_corpus, max_size=config["model"]["zh_vocab_size"])

gloss_vocab.save(OUTPUT_DIR / "gloss_vocab.json")
zh_vocab.save(OUTPUT_DIR / "zh_vocab.json")

config["model"]["gloss_vocab_size"] = len(gloss_vocab)
config["model"]["zh_vocab_size"] = len(zh_vocab)
config["save_dir"] = OUTPUT_DIR.as_posix()
with (OUTPUT_DIR / "runtime_config.yaml").open("w", encoding="utf-8") as file:
    yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

train_dataset = GlossChineseDataset(
    tsv_path=train_path.as_posix(),
    gloss_vocab=gloss_vocab,
    zh_vocab=zh_vocab,
    max_gloss_len=config["data"]["max_gloss_len"],
    max_zh_len=config["data"]["max_zh_len"],
)
val_dataset = GlossChineseDataset(
    tsv_path=val_path.as_posix(),
    gloss_vocab=gloss_vocab,
    zh_vocab=zh_vocab,
    max_gloss_len=config["data"]["max_gloss_len"],
    max_zh_len=config["data"]["max_zh_len"],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=True,
    collate_fn=GlossChineseDataset.collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=False,
    collate_fn=GlossChineseDataset.collate_fn,
)

encoder = GlossEncoder(
    gloss_vocab_size=config["model"]["gloss_vocab_size"],
    embed_dim=config["model"]["embed_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    num_layers=config["model"]["num_layers"],
    dropout=config["model"]["dropout"],
)
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
print(f"训练完成，最佳验证损失：{result['best_val_loss']:.4f}，最佳 BLEU-4：{result['best_bleu4']:.2f}")
print(f"最佳模型路径：{result['best_model_path']}")
PY