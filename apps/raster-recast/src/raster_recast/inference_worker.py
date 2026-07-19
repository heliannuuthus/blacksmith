"""Standalone FLUX-Text/MMGP inference worker with local Unix-socket IPC."""
# mypy: disable-error-code="import-not-found,import-untyped"

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

AUTH_KEY = b"RasterRecast-FLUX-Text"
DEFAULT_TRANSFORMER = "flux1_fill_dev_transformer_mmgp.safetensors"
DEFAULT_TEXT_ENCODER = "flux1_fill_dev_text_encoder_2_mmgp.safetensors"
DEFAULT_LORA = "flux-text-multisize.safetensors"


class FluxTextEngine:
    """Long-lived, directly loaded FLUX-Text pipeline using MMGP offload."""

    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root
        self.models = runtime_root / "models"
        self.source = runtime_root / "fluxtext-runtime"
        self.pipe: Any | None = None
        self._utils: Any | None = None
        self._condition_class: Any | None = None
        self._generate_fill: Any | None = None

    def _path(self, category: str, environment_name: str, default: str) -> Path:
        path = self.models / category / os.environ.get(environment_name, default)
        if not path.is_file():
            raise FileNotFoundError(f"缺少模型文件：{path}")
        return path

    def load(self) -> None:
        """Load and retain the quantized pipeline without importing ComfyUI."""
        if self.pipe is not None:
            return

        sys.path.insert(0, str(self.source))
        utils_spec = importlib.util.spec_from_file_location(
            "raster_recast_fluxtext_utils", self.source / "utils.py"
        )
        if utils_spec is None or utils_spec.loader is None:
            raise RuntimeError("无法加载 FLUX-Text 字形模块")
        utils = importlib.util.module_from_spec(utils_spec)
        utils_spec.loader.exec_module(utils)

        import torch
        import yaml
        from diffusers import FluxFillPipeline, FluxTransformer2DModel
        from flux_text.condition import Condition
        from flux_text.generate_fill import generate_fill
        from mmgp import offload
        from mmgp.offload import fast_load_transformers_model, profile_type
        from peft import LoraConfig
        from safetensors.torch import load_file
        from transformers import T5EncoderModel

        if not torch.cuda.is_available():
            raise RuntimeError("FLUX-Text 需要可用的 NVIDIA CUDA 显卡")

        transformer_path = self._path(
            "diffusion_models", "RASTER_RECAST_FLUX_TRANSFORMER", DEFAULT_TRANSFORMER
        )
        text_encoder_path = self._path(
            "text_encoders", "RASTER_RECAST_FLUX_TEXT_ENCODER", DEFAULT_TEXT_ENCODER
        )
        lora_path = self._path("loras", "RASTER_RECAST_FLUX_TEXT_LORA", DEFAULT_LORA)
        base_path = self.models / "AIFSH" / "flux1-fill-dev-mmgp"
        if not (base_path / "model_index.json").is_file():
            raise FileNotFoundError(f"缺少 FLUX.1 Fill 基础组件：{base_path}")

        transformer = fast_load_transformers_model(
            model_path=str(transformer_path),
            modelClass=FluxTransformer2DModel,
            do_quantize=True,
            forcedConfigPath=str(self.source / "configs" / "transformer_config.json"),
        )
        text_encoder_2 = fast_load_transformers_model(
            model_path=str(text_encoder_path),
            modelClass=T5EncoderModel,
            forcedConfigPath=str(self.source / "configs" / "text_encoder_2.json"),
        )
        pipe = FluxFillPipeline.from_pretrained(
            str(base_path),
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        )
        pipe.vae_scale_factor = 8
        pipe.text_encoder.requires_grad_(False).eval()
        pipe.text_encoder_2.requires_grad_(False).eval()
        pipe.vae.requires_grad_(False).eval()

        with (self.source / "config.yaml").open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        pipe.transformer.add_adapter(LoraConfig(**config["train"]["lora_config"]))
        state_dict = load_file(str(lora_path))
        normalized = {
            key.replace("lora_A", "lora_A.default")
            .replace("lora_B", "lora_B.default")
            .replace("transformer.", ""): value
            for key, value in state_dict.items()
        }
        pipe.transformer.load_state_dict(normalized, strict=False)
        del state_dict, normalized
        offload.profile(pipe, profile_type.LowRAM_LowVRAM)

        self.pipe = pipe
        self._utils = utils
        self._condition_class = Condition
        self._generate_fill = generate_fill

    def edit(
        self,
        image_bytes: bytes,
        replacement: str,
        prompt: str,
        steps: int,
        seed: int,
    ) -> bytes:
        """Generate a contextual replacement crop and encode it as PNG."""
        self.load()
        assert self.pipe is not None
        assert self._utils is not None
        assert self._condition_class is not None
        assert self._generate_fill is not None

        import numpy as np
        import torch

        with Image.open(io.BytesIO(image_bytes)) as loaded:
            rgba = loaded.convert("RGBA")
        image = rgba.convert("RGB")
        mask = ImageOps.invert(rgba.getchannel("A")).convert("RGB")
        glyph = self._utils.render_glyph_multi(image, mask, [replacement])

        original_width, original_height = image.size
        num_pixel = min(
            self._utils.PIXELS,
            key=lambda value: abs(value - original_width * original_height),
        )
        ratios = self._utils.get_aspect_ratios_dict(num_pixel)
        closest = self._utils.get_closest_ratio(
            original_height,
            original_width,
            self._utils.ASPECT_RATIO_LD_LIST,
        )
        target_height, target_width = ratios[closest]

        hint = np.array(mask.resize((target_width, target_height))) / 255
        condition_image = np.array(glyph.resize((target_width, target_height)).convert("RGB"))
        condition_image = (255 - condition_image) / 255
        resized = image.resize((target_width, target_height))
        condition = self._condition_class(
            condition_type="word_fill",
            condition=[condition_image, hint, resized],
            position_delta=[0, 0],
        )
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self._generate_fill(
            self.pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_height,
            width=target_width,
            generator=generator,
            config_path=str(self.source / "config.yaml"),
            num_inference_steps=steps,
        ).images[0]
        output = io.BytesIO()
        result.save(output, format="PNG")
        return output.getvalue()


def serve(socket_path: Path, runtime_root: Path) -> None:
    """Serve one local GUI over a private Unix socket until terminated."""
    socket_path.unlink(missing_ok=True)
    engine = FluxTextEngine(runtime_root)
    listener = Listener(str(socket_path), family="AF_UNIX", authkey=AUTH_KEY)
    socket_path.chmod(0o600)
    try:
        while True:
            connection = listener.accept()
            try:
                request = connection.recv()
                operation = request.get("operation")
                if operation == "health":
                    connection.send({"type": "ready"})
                elif operation == "shutdown":
                    connection.send({"type": "stopping"})
                    return
                elif operation == "edit":
                    connection.send({"type": "progress", "message": "正在加载 FLUX-Text 模型…"})
                    engine.load()
                    connection.send({"type": "progress", "message": "正在进行 AI 字形重建…"})
                    result = engine.edit(
                        request["image"],
                        request["replacement"],
                        request["prompt"],
                        request["steps"],
                        request["seed"],
                    )
                    connection.send({"type": "result", "image": result})
                else:
                    raise ValueError(f"未知操作：{operation}")
            except Exception as exc:
                connection.send(
                    {
                        "type": "error",
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                connection.close()
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    arguments = parser.parse_args()
    serve(arguments.socket, arguments.runtime_root)


if __name__ == "__main__":
    main()
