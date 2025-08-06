MODELS_CONFIG = {
    # tiny model for fast inference
    "tiny": {
        "model_id": "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M",
        "processor_id": "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M",  # 添加processor信息
        "onnx_path": "onnx_models/clip_tiny.onnx",
    },
    # base model with smaller footprint
    "small": {
        "model_id": "openai/clip-vit-base-patch32",
        "processor_id": "openai/clip-vit-base-patch32",
        "onnx_path": "onnx_models/clip_small.onnx",
    },
    # base model with higher resolution and accuracy
    "large": {
        "model_id": "openai/clip-vit-base-patch16",
        "processor_id": "openai/clip-vit-base-patch16",
        "onnx_path": "onnx_models/clip_large.onnx",
    },
    # 可以继续添加其他CLIP模型
}
