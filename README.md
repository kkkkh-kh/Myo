# Chinese Sign Language Gloss Sequence -> Natural Chinese Sentence

这是一个完整的 Python 项目，用于把中文手语 Gloss 序列翻译为自然中文句子，并重点面向 **INT8 ONNX Runtime 部署** 与 **60MB 峰值推理内存约束**。

项目实现包含：

- 基于 BiGRU + Bahdanau Attention 的 Seq2Seq 模型
- 规则驱动的 Gloss 预排序模块
- 中文后处理模块（`了`、量词、`的`、标点规范化）
- 训练、验证、BLEU-4 / ROUGE-L / WER 评估
- ONNX 导出、INT8 动态量化、内存测试
- 命令行推理与批量翻译
- 可自举的基础测试用例

## 1. 环境与依赖兼容性

本项目默认使用 **Conda 环境**，推荐固定到 **Python 3.10**。

当前依赖版本组合：

- `torch==2.1.0`
- `numpy==1.24.0`
- `onnxruntime==1.16.0`

与 **Python 3.12 / 3.13** 不匹配，容易出现安装失败、缺少预编译 wheel 或运行时报错。因此项目现在统一使用 Conda，并将安装解释器范围收紧为：

```text
python >= 3.10, < 3.12
```

依赖声明的对应关系如下：

- `requirements.txt`：保存精确 pip 依赖版本
- `environment.yml`：Conda 创建环境的标准入口
- `setup.py`：直接读取 `requirements.txt`，避免重复维护另一份依赖列表

## 2. 安装步骤

推荐安装方式：

```bash
cd sign_language_translator
conda env create -f environment.yml
conda activate sign-language-translator
pip install -e . --no-deps
```

如果环境已经存在，可更新：

```bash
conda env update -f environment.yml --prune
conda activate sign-language-translator
pip install -e . --no-deps
```

说明：

- `environment.yml` 已包含运行所需的全部依赖版本
- `pip install -e . --no-deps` 只安装项目本身，避免再次解析依赖导致版本漂移
- 不再建议把 `venv + pip install -r requirements.txt` 作为默认安装方式

## 3. 数据准备

项目训练输入现在同时支持两种格式：

1. 两列 TSV：`gloss_sequence<TAB>chinese_sentence`
2. 你当前使用的五列表头 CSV：`Number, Translator, Chinese Sentences, Gloss, Note`

TSV 示例：

```text
我 昨天 买 苹果	我昨天买了苹果
残疾人 申请 政府 补偿	残疾人向政府申请补偿
```

CSV 示例表头：

```text
Number,Translator,Chinese Sentences,Gloss,Note
train-00001,A,2023年高考到了。,2/0/2/3/高/考/时间/到/。, 
```

当前代码会自动读取 CSV 中的 `Chinese Sentences` 和 `Gloss` 两列，并把 Gloss 里的 `/` 自动转换成空格分词形式。

### 推荐数据来源

- `CE-CSL`：适合补充正式表达、教育与日常语料
- `CSL-Daily`：适合构建日常场景 gloss -> 中文翻译数据

### 推荐准备流程

1. 将原始数据整理成两列 TSV，或者保留当前五列表头 CSV。
2. 使用 `data/preprocess.py` 做清洗与分词。
3. 使用 `data/vocabulary.py` 构建 gloss 词表与中文词表。
4. 将训练、验证、测试集保存为以下任一命名组合。

推荐目录：

```text
datasets/
├── train.tsv 或 train.csv
├── val.tsv 或 dev.csv
└── test.tsv 或 test.csv
```

## 4. 训练命令

默认配置写在 `configs/default.yaml` 中，推荐直接使用项目内置脚本：

```bash
bash scripts/train.sh
```

如果数据目录不在默认位置，可通过环境变量覆盖：

```bash
DATA_DIR=./datasets OUTPUT_DIR=./checkpoints bash scripts/train.sh
```

脚本会自动完成以下步骤：

- 自动读取 `train.tsv/train.csv` 与 `val.tsv/dev.csv`
- 构建 gloss / 中文词表
- 保存 `gloss_vocab.json` 与 `zh_vocab.json`
- 训练模型并执行验证
- 输出 `best_model.pt` 与 `training_log.csv`

## 5. 导出与量化

```bash
bash scripts/deploy.sh
```

执行完成后，`checkpoints/` 目录下会生成：

- `encoder.onnx`
- `decoder.onnx`
- `encoder.int8.onnx`
- `decoder.int8.onnx`
- `gloss_vocab.json`
- `zh_vocab.json`

## 6. 推理命令

单句推理：

```bash
python inference/translate.py --gloss "我 昨天 买 苹果"
# 输出: 我昨天买了苹果
```

批量推理：

```bash
python inference/translate.py --file input.txt --output output.txt
```

交互模式：

```bash
python inference/translate.py --interactive
```

带内存检测的推理：

```bash
python inference/translate.py --gloss "残疾人 申请 政府 补偿" --memory_check
```

## 7. 内存基准

下表给出推荐部署形态与目标约束：

| 部署形态 | 模型格式 | 峰值推理内存 | 单句延迟 | 说明 |
|--------|--------|-------------|---------|------|
| 训练态基线 | PyTorch FP32 | > 60MB | 较高 | 仅用于训练与验证 |
| 部署推荐 | ONNX INT8 | <= 60MB | <= 200ms | 本项目默认目标 |
| 测试演示 | ONNX INT8 最小词表 | 通常远低于 60MB | 低 | 用于 CI / 单元测试 |

## 8. BLEU-4 基准

下表给出建议评估目标：

| 数据集 | 指标 | 目标值 |
|------|------|-------|
| CSL-Daily 测试集 | BLEU-4 | >= 20 |
| CSL-Daily 测试集 | ROUGE-L | >= 40 |
| CSL-Daily 测试集 | WER | <= 25% |

## 9. 项目结构

```text
sign_language_translator/
├── configs/
├── data/
├── deploy/
├── environment.yml
├── inference/
├── model/
├── modules/
├── tests/
├── train/
├── README.md
├── requirements.txt
└── setup.py
```

## 10. 测试

可运行以下命令做基础检查：

```bash
python -m unittest discover tests
```

如需重点验证内存限制：

```bash
python -m unittest tests.test_memory
```

## 11. 说明

- 代码注释统一使用英文。
- 终端日志与打印信息统一使用中文。
- 推理流程默认优先加载 `*.int8.onnx`，如不存在则回退到 FP32 ONNX。
- `tests/` 中的端到端测试会自动导出并量化一个最小可运行模型，以保证仓库开箱可验证。
## 12. 早停与续训（新增）

当前训练流程已支持：
- 早停（Early Stopping）
- 断点续训（Resume Training）

### 12.1 早停配置

在 `configs/default.yaml` 的 `train` 段中：

```yaml
train:
  early_stopping_patience: 5
  early_stopping_min_delta: 0.0
```

说明：
- `early_stopping_patience`：验证集连续多少个 epoch 无提升后停止。
- `early_stopping_min_delta`：判定“提升”所需的最小下降幅度（针对 `val_loss`）。

### 12.2 检查点文件

训练过程中会产出两个关键文件：
- `best_model.pt`：验证集最优模型。
- `latest_model.pt`：最近 epoch 的完整训练状态（包含模型参数、优化器、调度器、早停状态），推荐用于续训。

### 12.3 使用 `scripts/train.py` 续训

从最新断点自动续训：

```bash
cd sign_language_translator
CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR=./checkpoints/word_order_mix AUTO_RESUME=1 python scripts/train.py
```

从指定断点续训：

```bash
cd sign_language_translator
CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR=./checkpoints/word_order_mix RESUME_FROM=./checkpoints/word_order_mix/latest_model.pt python scripts/train.py
```

也可在配置中指定（优先级低于环境变量）：

```yaml
train:
  resume_from: null
```

### 12.4 使用 `scripts/train.sh` 续训

`scripts/train.sh` 现支持以下变量：
- `RESUME_FROM`：指定 checkpoint 路径。
- `AUTO_RESUME=1`：自动读取 `OUTPUT_DIR/latest_model.pt`。
- `SKIP_AUGMENT=1`：续训时跳过数据增强预处理步骤。

示例：

```bash
cd sign_language_translator
CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR=./checkpoints/word_order_mix SKIP_AUGMENT=1 RESUME_FROM=./checkpoints/word_order_mix/latest_model.pt bash scripts/train.sh
```

### 12.5 续训建议

- 优先用 `latest_model.pt` 续训，能完整恢复优化器与学习率调度状态。
- 若仅有 `best_model.pt`，也可继续训练，但优化器状态可能无法完整恢复。
- 续训时建议保持与上次训练一致的 `OUTPUT_DIR`、词表和配置。

## 13. Early Stopping and Resume (Plain ASCII)

Training now supports both early stopping and checkpoint resume.

### Config keys

In `configs/default.yaml`:

```yaml
train:
  early_stopping_patience: 5
  early_stopping_min_delta: 0.0
  resume_from: null
```

- `early_stopping_patience`: stop when validation loss does not improve for N epochs.
- `early_stopping_min_delta`: minimum `val_loss` decrease to count as an improvement.
- `resume_from`: optional checkpoint path from config (env vars still override).

### Checkpoints

- `best_model.pt`: best validation checkpoint.
- `latest_model.pt`: latest full training state (model + optimizer + scheduler + patience), recommended for resume.

### Resume with `scripts/train.py`

Auto-resume from `OUTPUT_DIR/latest_model.pt`:

```bash
cd sign_language_translator
CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR=./checkpoints/word_order_mix AUTO_RESUME=1 python scripts/train.py
```

Resume from explicit checkpoint:

```bash
cd sign_language_translator
CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR=./checkpoints/word_order_mix RESUME_FROM=./checkpoints/word_order_mix/latest_model.pt python scripts/train.py
```

### Resume with `scripts/train.sh`

Supported env vars:
- `RESUME_FROM`
- `AUTO_RESUME=1`
- `SKIP_AUGMENT=1`

Example:

```bash
cd sign_language_translator
CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR=./checkpoints/word_order_mix SKIP_AUGMENT=1 RESUME_FROM=./checkpoints/word_order_mix/latest_model.pt bash scripts/train.sh
```
