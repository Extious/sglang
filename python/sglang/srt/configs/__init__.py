import importlib


_CONFIG_IMPORTS = {
    "AfmoeConfig": "sglang.srt.configs.afmoe",
    "BailingHybridConfig": "sglang.srt.configs.bailing_hybrid",
    "ChatGLMConfig": "sglang.srt.configs.chatglm",
    "DbrxConfig": "sglang.srt.configs.dbrx",
    "DeepseekVL2Config": "sglang.srt.configs.deepseekvl2",
    "DotsOCRConfig": "sglang.srt.configs.dots_ocr",
    "DotsVLMConfig": "sglang.srt.configs.dots_vlm",
    "ExaoneConfig": "sglang.srt.configs.exaone",
    "FalconH1Config": "sglang.srt.configs.falcon_h1",
    "GraniteMoeHybridConfig": "sglang.srt.configs.granitemoehybrid",
    "MultiModalityConfig": "sglang.srt.configs.janus_pro",
    "JetNemotronConfig": "sglang.srt.configs.jet_nemotron",
    "JetVLMConfig": "sglang.srt.configs.jet_vlm",
    "KimiK25Config": "sglang.srt.configs.kimi_k25",
    "KimiLinearConfig": "sglang.srt.configs.kimi_linear",
    "KimiVLConfig": "sglang.srt.configs.kimi_vl",
    "MoonViTConfig": "sglang.srt.configs.kimi_vl_moonvit",
    "Lfm2Config": "sglang.srt.configs.lfm2",
    "Lfm2MoeConfig": "sglang.srt.configs.lfm2_moe",
    "Lfm2VlConfig": "sglang.srt.configs.lfm2_vl",
    "LongcatFlashConfig": "sglang.srt.configs.longcat_flash",
    "NemotronH_Nano_VL_V2_Config": "sglang.srt.configs.nano_nemotron_vl",
    "NemotronHConfig": "sglang.srt.configs.nemotron_h",
    "Olmo3Config": "sglang.srt.configs.olmo3",
    "Qwen3_5Config": "sglang.srt.configs.qwen3_5",
    "Qwen3_5MoeConfig": "sglang.srt.configs.qwen3_5",
    "Qwen3NextConfig": "sglang.srt.configs.qwen3_next",
    "Step3TextConfig": "sglang.srt.configs.step3_vl",
    "Step3VisionEncoderConfig": "sglang.srt.configs.step3_vl",
    "Step3VLConfig": "sglang.srt.configs.step3_vl",
    "Step3p5Config": "sglang.srt.configs.step3p5",
}

__all__ = list(_CONFIG_IMPORTS)


def __getattr__(name: str):
    module_name = _CONFIG_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
