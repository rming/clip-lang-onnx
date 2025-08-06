import torch
from transformers import CLIPModel
from config import MODELS_CONFIG


def export_clip_to_onnx(
    model_id="wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M",
    onnx_path="onnx_models/tinyclip.onnx",
):
    """
    将CLIP模型导出为ONNX格式

    Args:
        model_id (str): HuggingFace模型ID
        onnx_path (str): 输出的ONNX文件路径
    """
    print(f"Loading model: {model_id}")
    model = CLIPModel.from_pretrained(model_id)
    model.eval()

    dummy_input = {
        "input_ids": torch.randint(0, 1000, (1, 16)),
        "pixel_values": torch.randn(1, 3, 224, 224),
        "attention_mask": torch.ones(1, 16, dtype=torch.long),
    }

    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        args=(),
        kwargs=dummy_input,
        f=onnx_path,
        input_names=["input_ids", "pixel_values", "attention_mask"],
        output_names=["logits_per_image", "logits_per_text"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "pixel_values": {0: "batch"},
            "logits_per_image": {0: "batch"},
            "logits_per_text": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"✓ Model exported successfully to {onnx_path}")


if __name__ == "__main__":
    print("可用的模型别名:")
    for alias, config in MODELS_CONFIG.items():
        print(f"  {alias}: {config['model_id']}")

    try:
        model_alias = input("请输入模型别名: ").strip().lower()

        if model_alias in MODELS_CONFIG:
            selected_model = MODELS_CONFIG[model_alias]
            print(f"开始导出模型: {selected_model['model_id']}")
            try:
                export_clip_to_onnx(
                    selected_model["model_id"], selected_model["onnx_path"]
                )
                print("模型导出完成！")
            except Exception as e:
                print(f"✗ 导出失败: {e}")
        else:
            print(f"✗ 未找到模型别名 '{model_alias}'")
            print(f"可用别名: {', '.join(MODELS_CONFIG.keys())}")

    except KeyboardInterrupt:
        print("\n程序已取消。")
