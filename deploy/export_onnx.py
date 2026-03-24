# Module description: ONNX export utilities for encoder-decoder deployment artifacts.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import onnx
import torch
import yaml
from torch import nn

from data.vocabulary import Vocabulary
from model.decoder import ChineseDecoder
from model.encoder import GlossEncoder
from model.seq2seq import Seq2Seq


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


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_config_path(save_path: Path, config_path: Optional[str] = None) -> Path:
    if config_path is not None:
        return Path(config_path)
    runtime_config = save_path / "runtime_config.yaml"
    if runtime_config.exists():
        return runtime_config
    return _project_root() / "configs" / "default.yaml"


def _resolve_checkpoint_path(save_path: Path, config: Dict) -> Path:
    deploy_config = config.get("deploy", {})
    distill_config = config.get("distillation", {})
    distilled_candidate = save_path / Path(distill_config.get("save_path", "distilled_model.pt")).name
    best_candidate = save_path / "best_model.pt"

    if deploy_config.get("use_distilled", True) and distilled_candidate.exists():
        return distilled_candidate
    if best_candidate.exists():
        return best_candidate
    if distilled_candidate.exists():
        return distilled_candidate
    raise FileNotFoundError(
        f"No exportable checkpoint found under {save_path}. Expected {distilled_candidate.name} or {best_candidate.name}."
    )


def load_model_for_export(
    save_dir: str,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Tuple[Seq2Seq, Path, Dict]:
    """Load the checkpoint selected for ONNX export.

    The function prefers ``distilled_model.pt`` when ``deploy.use_distilled`` is
    enabled and the file exists; otherwise it falls back to ``best_model.pt``.
    """
    save_path = Path(save_dir)
    config_file = _resolve_config_path(save_path, config_path=config_path)
    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    gloss_vocab = Vocabulary.load(save_path / "gloss_vocab.json")
    zh_vocab = Vocabulary.load(save_path / "zh_vocab.json")
    config.setdefault("model", {})
    config.setdefault("encoder", {})
    config["model"]["gloss_vocab_size"] = len(gloss_vocab)
    config["model"]["zh_vocab_size"] = len(zh_vocab)

    encoder_config = config.get("encoder", {})
    encoder = GlossEncoder(
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
    decoder = ChineseDecoder(
        zh_vocab_size=config["model"]["zh_vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )
    model = Seq2Seq(encoder=encoder, decoder=decoder)

    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else _resolve_checkpoint_path(save_path, config)
    state = torch.load(resolved_checkpoint.as_posix(), map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    print(f"当前导出的权重文件: {resolved_checkpoint}")
    return model, resolved_checkpoint, config


def export_to_onnx(
    model: Optional[nn.Module],
    save_dir: str,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Export encoder and decoder artifacts to ONNX.

    Args:
        model: In-memory PyTorch model. When ``None``, the function automatically
            loads the preferred checkpoint from ``save_dir``.
        save_dir: Directory where vocabularies, checkpoints, and exported ONNX files live.
        config_path: Optional config path used when the model needs to be loaded.
        checkpoint_path: Optional explicit checkpoint path. When omitted, the
            function prefers ``distilled_model.pt`` over ``best_model.pt``.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if model is None:
        model, _, _ = load_model_for_export(save_dir=save_dir, config_path=config_path, checkpoint_path=checkpoint_path)
    elif checkpoint_path is not None:
        print(f"当前导出的权重文件: {Path(checkpoint_path)}")
    else:
        print("当前导出的权重文件: <in-memory model>")

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
