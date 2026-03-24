from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _session(model_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])


def _verify_encoder(model_path: Path) -> Dict[str, tuple]:
    session = _session(model_path)
    outputs = session.run(None, {"input_ids": np.ones((1, 8), dtype=np.int64)})
    return {meta.name: output.shape for meta, output in zip(session.get_outputs(), outputs)}


def _verify_decoder(model_path: Path) -> Dict[str, tuple]:
    session = _session(model_path)
    inputs = {
        "input_token": np.ones((1,), dtype=np.int64),
        "hidden": np.zeros((2, 1, 256), dtype=np.float32),
        "enc_output": np.zeros((1, 8, 512), dtype=np.float32),
        "src_mask": np.ones((1, 8), dtype=np.bool_),
    }
    outputs = session.run(None, inputs)
    return {meta.name: output.shape for meta, output in zip(session.get_outputs(), outputs)}


def quantize_models(model_dir: str, output_dir: str) -> None:
    source_dir = Path(model_dir)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for name in ["encoder", "decoder"]:
        source_path = source_dir / f"{name}.onnx"
        target_path = target_dir / f"{name}.int8.onnx"
        quantize_dynamic(
            model_input=source_path.as_posix(),
            model_output=target_path.as_posix(),
            weight_type=QuantType.QInt8,
        )
        print(
            f"{name} 量化完成：FP32 {_size_mb(source_path):.2f} MB -> INT8 {_size_mb(target_path):.2f} MB"
        )

    encoder_shapes = _verify_encoder(target_dir / "encoder.int8.onnx")
    decoder_shapes = _verify_decoder(target_dir / "decoder.int8.onnx")
    print(f"编码器量化模型校验通过：{encoder_shapes}")
    print(f"解码器量化模型校验通过：{decoder_shapes}")
