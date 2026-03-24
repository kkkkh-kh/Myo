from pathlib import Path
from typing import Tuple

import onnx
import torch
from torch import nn


class EncoderExportWrapper(nn.Module):
    """Wrap the encoder for ONNX export."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(input_ids)


class DecoderExportWrapper(nn.Module):
    """Wrap one decoder step for ONNX export."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decoder.forward_step(input_token, hidden, enc_output, src_mask)


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def export_to_onnx(model, save_dir: str) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    encoder_wrapper = EncoderExportWrapper(model.encoder).eval()
    decoder_wrapper = DecoderExportWrapper(model.decoder).eval()

    device = next(model.parameters()).device
    dummy_gloss = torch.ones(1, 8, dtype=torch.long, device=device)
    dummy_enc_output, dummy_enc_hidden = model.encoder(dummy_gloss)
    dummy_decoder_hidden = model.decoder.init_hidden(dummy_enc_hidden)
    dummy_input_token = torch.ones(1, dtype=torch.long, device=device)
    dummy_mask = dummy_gloss.ne(model.pad_id)

    encoder_path = save_path / "encoder.onnx"
    decoder_path = save_path / "decoder.onnx"

    torch.onnx.export(
        encoder_wrapper,
        (dummy_gloss,),
        encoder_path.as_posix(),
        input_names=["input_ids"],
        output_names=["enc_output", "enc_hidden"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "src_len"},
            "enc_output": {0: "batch", 1: "src_len"},
            "enc_hidden": {0: "batch"},
        },
        opset_version=17,
    )

    torch.onnx.export(
        decoder_wrapper,
        (dummy_input_token, dummy_decoder_hidden, dummy_enc_output, dummy_mask),
        decoder_path.as_posix(),
        input_names=["input_token", "hidden", "enc_output", "src_mask"],
        output_names=["logits", "next_hidden", "attn_weights"],
        dynamic_axes={
            "input_token": {0: "batch"},
            "hidden": {1: "batch"},
            "enc_output": {0: "batch", 1: "src_len"},
            "src_mask": {0: "batch", 1: "src_len"},
            "logits": {0: "batch"},
            "next_hidden": {1: "batch"},
            "attn_weights": {0: "batch", 1: "src_len"},
        },
        opset_version=17,
    )

    onnx.checker.check_model(onnx.load(encoder_path.as_posix()))
    onnx.checker.check_model(onnx.load(decoder_path.as_posix()))

    print(f"编码器 ONNX 导出完成，大小 {_size_mb(encoder_path):.2f} MB")
    print(f"解码器 ONNX 导出完成，大小 {_size_mb(decoder_path):.2f} MB")
