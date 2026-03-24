import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from deploy.memory_check import MemoryProfiler
from inference.pipeline import TranslationPipeline


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_MODELS = PROJECT_ROOT / "checkpoints"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gloss 到中文句子的翻译工具")
    parser.add_argument("--gloss", type=str, help="单条 gloss 输入")
    parser.add_argument("--file", type=str, help="批量输入文件，每行一条 gloss")
    parser.add_argument("--output", type=str, help="批量输出文件")
    parser.add_argument("--interactive", action="store_true", help="进入交互模式")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODELS.as_posix(), help="量化模型目录")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix(), help="配置文件路径")
    parser.add_argument("--memory_check", action="store_true", help="翻译前执行内存检测")
    return parser


def _run_memory_check(pipeline: TranslationPipeline, seed_inputs) -> None:
    profiler = MemoryProfiler()
    result = profiler.measure_inference_memory(pipeline, seed_inputs)
    print(
        "内存检测完成：峰值 {peak_mb:.2f}MB，平均 {mean_mb:.2f}MB，标准差 {std_mb:.2f}MB，超限次数 {passes_over_limit}".format(
            **result
        )
    )
    profiler.assert_under_limit(result["peak_mb"], pipeline.memory_limit_mb)


def _translate_file(pipeline: TranslationPipeline, input_path: str, output_path: str = None) -> None:
    with open(input_path, "r", encoding="utf-8") as file:
        gloss_list = [line.strip() for line in file if line.strip()]
    results = pipeline.batch_translate(gloss_list)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write("\n".join(results) + ("\n" if results else ""))
        print(f"批量翻译完成，结果已写入：{output_path}")
    else:
        for line in results:
            print(line)


def _interactive_loop(pipeline: TranslationPipeline) -> None:
    print("进入交互模式，输入空行即可退出。")
    while True:
        gloss = input("请输入 gloss: ").strip()
        if not gloss:
            print("已退出交互模式。")
            break
        print(pipeline.translate(gloss))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not any([args.gloss, args.file, args.interactive]):
        parser.error("请至少提供 --gloss、--file 或 --interactive 之一")

    pipeline = TranslationPipeline(model_dir=args.model_dir, config_path=args.config)

    if args.memory_check:
        if args.gloss:
            seed_inputs = [args.gloss]
        elif args.file:
            with open(args.file, "r", encoding="utf-8") as file:
                seed_inputs = [line.strip() for line in file if line.strip()][:50]
        else:
            seed_inputs = ["我 昨天 买 苹果", "残疾人 申请 政府 补偿", "补偿 残疾人 申请"]
        _run_memory_check(pipeline, seed_inputs)

    if args.gloss:
        print(pipeline.translate(args.gloss))
    if args.file:
        _translate_file(pipeline, args.file, args.output)
    if args.interactive:
        _interactive_loop(pipeline)


if __name__ == "__main__":
    main()
