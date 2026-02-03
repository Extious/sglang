from __future__ import annotations

from importlib import import_module
from typing import Any


def _optional_import(module: str, name: str) -> Any:
    try:
        obj = getattr(import_module(module), name)
    except Exception as e:
        obj = type(
            name,
            (),
            {
                "__sglang_optional_import_failed__": True,
                "__sglang_import_error__": repr(e),
            },
        )
        obj.__module__ = __name__
    globals()[name] = obj
    return obj


ChatGLMConfig = _optional_import("sglang.srt.configs.chatglm", "ChatGLMConfig")
DbrxConfig = _optional_import("sglang.srt.configs.dbrx", "DbrxConfig")
DeepseekVL2Config = _optional_import("sglang.srt.configs.deepseekvl2", "DeepseekVL2Config")
DotsOCRConfig = _optional_import("sglang.srt.configs.dots_ocr", "DotsOCRConfig")
DotsVLMConfig = _optional_import("sglang.srt.configs.dots_vlm", "DotsVLMConfig")
ExaoneConfig = _optional_import("sglang.srt.configs.exaone", "ExaoneConfig")
FalconH1Config = _optional_import("sglang.srt.configs.falcon_h1", "FalconH1Config")
MultiModalityConfig = _optional_import("sglang.srt.configs.janus_pro", "MultiModalityConfig")
JetNemotronConfig = _optional_import("sglang.srt.configs.jet_nemotron", "JetNemotronConfig")
JetVLMConfig = _optional_import("sglang.srt.configs.jet_vlm", "JetVLMConfig")
KimiLinearConfig = _optional_import("sglang.srt.configs.kimi_linear", "KimiLinearConfig")
KimiVLConfig = _optional_import("sglang.srt.configs.kimi_vl", "KimiVLConfig")
MoonViTConfig = _optional_import("sglang.srt.configs.kimi_vl_moonvit", "MoonViTConfig")
LongcatFlashConfig = _optional_import("sglang.srt.configs.longcat_flash", "LongcatFlashConfig")
NemotronH_Nano_VL_V2_Config = _optional_import("sglang.srt.configs.nano_nemotron_vl", "NemotronH_Nano_VL_V2_Config")
NemotronHConfig = _optional_import("sglang.srt.configs.nemotron_h", "NemotronHConfig")
Olmo3Config = _optional_import("sglang.srt.configs.olmo3", "Olmo3Config")
Qwen3NextConfig = _optional_import("sglang.srt.configs.qwen3_next", "Qwen3NextConfig")
Step3TextConfig = _optional_import("sglang.srt.configs.step3_vl", "Step3TextConfig")
Step3VisionEncoderConfig = _optional_import("sglang.srt.configs.step3_vl", "Step3VisionEncoderConfig")
Step3VLConfig = _optional_import("sglang.srt.configs.step3_vl", "Step3VLConfig")

__all__ = [
    "ExaoneConfig",
    "ChatGLMConfig",
    "DbrxConfig",
    "DeepseekVL2Config",
    "LongcatFlashConfig",
    "MultiModalityConfig",
    "KimiVLConfig",
    "MoonViTConfig",
    "Step3VLConfig",
    "Step3TextConfig",
    "Step3VisionEncoderConfig",
    "Olmo3Config",
    "KimiLinearConfig",
    "Qwen3NextConfig",
    "DotsVLMConfig",
    "DotsOCRConfig",
    "FalconH1Config",
    "NemotronHConfig",
    "NemotronH_Nano_VL_V2_Config",
    "JetNemotronConfig",
    "JetVLMConfig",
]
