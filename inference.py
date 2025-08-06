import argparse
import os
import glob
import numpy as np
import onnxruntime as ort
from transformers import CLIPProcessor
from PIL import Image
from config import MODELS_CONFIG


def get_processor_from_onnx_path(onnx_path):
    """
    根据ONNX模型路径自动选择对应的CLIPProcessor

    Args:
        onnx_path (str): ONNX模型文件路径

    Returns:
        CLIPProcessor: 对应的CLIP处理器
    """
    # 规范化路径
    onnx_path = os.path.normpath(onnx_path)

    # 遍历配置，找到匹配的模型
    for model_name, config in MODELS_CONFIG.items():
        if os.path.normpath(config["onnx_path"]) in onnx_path or onnx_path.endswith(
            os.path.basename(config["onnx_path"])
        ):
            processor_id = config["processor_id"]
            print(f"自动选择的 CLIPProcessor: {processor_id} (模型: {model_name})")
            return CLIPProcessor.from_pretrained(processor_id)

    # 如果没有找到匹配的配置，使用默认的
    default_processor_id = "openai/clip-vit-base-patch16"
    print(f"未找到匹配的配置，使用默认 CLIPProcessor: {default_processor_id}")
    return CLIPProcessor.from_pretrained(default_processor_id)


def load_onnx_session(onnx_path):
    """
    加载ONNX推理会话

    Args:
        onnx_path (str): ONNX模型文件路径

    Returns:
        ort.InferenceSession: ONNX推理会话
    """
    # 优先使用GPU，如果不可用则使用CPU
    providers = (
        ["CUDAExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    print(f"Using providers: {providers}")

    session = ort.InferenceSession(onnx_path, providers=providers)
    return session


def predict_image_language(
    onnx_path, image_paths, processor=None, prompts=None, top_k=3
):
    """
    使用ONNX模型预测图像中的主要语言

    Args:
        onnx_path (str): ONNX模型文件路径
        image_paths (list): 图像文件路径列表
        processor: CLIP处理器，如果为None则根据onnx_path自动选择
        prompts (list): 文本提示列表，如果为None则使用默认语言检测提示
        top_k (int): 返回前k个结果
    """
    # 支持传入单条路径或多条路径
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # 自动选择 processor
    if processor is None:
        processor = get_processor_from_onnx_path(onnx_path)

    # 默认语言检测提示
    if prompts is None:
        prompts = [
            "The majority of the text in this image is Chinese.",
            "The majority of the text in this image is Japanese.",
            "The majority of the text in this image is Korean.",
            "The majority of the text in this image is English.",
            "The majority of the text in this image is Russian.",
        ]

    # 加载ONNX推理会话
    session = load_onnx_session(onnx_path)

    # 循环处理每张图
    for img_path in image_paths:
        try:
            # 检查文件是否存在
            if not os.path.exists(img_path):
                print(f"✗ 文件不存在: {img_path}")
                continue

            image = Image.open(img_path).convert("RGB")
            inputs = processor(
                images=image, text=prompts, return_tensors="np", padding=True
            )

            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "pixel_values": inputs["pixel_values"].astype(np.float32),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }

            # 执行推理
            logits_per_image = session.run(["logits_per_image"], ort_inputs)[0]

            # 计算softmax概率
            probs = np.exp(logits_per_image) / np.sum(
                np.exp(logits_per_image), axis=1, keepdims=True
            )

            # 创建结果字典
            data = dict(zip(prompts, probs[0].tolist()))

            # 获取top_k结果
            top_results = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_k]

            print(f"\nResults for {img_path}:")
            for i, (label, score) in enumerate(top_results, 1):
                # 简化标签显示
                lang = label.split()[-1].rstrip(".")
                print(f"  {i}. {lang}: {score:.4f}")

        except Exception as e:
            print(f"✗ 处理图像 {img_path} 时出错: {e}")


def get_image_files(path):
    """
    获取指定路径下的所有图像文件

    Args:
        path (str): 文件路径或目录路径

    Returns:
        list: 图像文件路径列表
    """
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        # 支持的图像格式
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(path, ext)))
            image_files.extend(glob.glob(os.path.join(path, ext.upper())))
        return sorted(image_files)
    return []


def main():
    parser = argparse.ArgumentParser(description="CLIP ONNX语言检测推理")
    parser.add_argument(
        "--model",
        default="onnx_models/clip_large.onnx",
        required=False,
        help="ONNX模型文件路径 (默认: onnx_models/clip_large.onnx)",
    )
    parser.add_argument(
        "--images", nargs="+", required=True, help="图像文件路径或目录路径"
    )
    parser.add_argument("--top-k", type=int, default=3, help="显示前k个结果 (默认: 3)")

    args = parser.parse_args()

    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"✗ 模型文件不存在: {args.model}")
        return

    # 收集所有图像文件
    all_images = []
    for path in args.images:
        images = get_image_files(path)
        if images:
            all_images.extend(images)
        else:
            print(f"⚠ 路径中没有找到图像文件: {path}")

    if not all_images:
        print("✗ 没有找到任何图像文件")
        return

    print(f"找到 {len(all_images)} 个图像文件")
    print(f"使用模型: {args.model}")

    # 执行推理
    predict_image_language(
        onnx_path=args.model, image_paths=all_images, top_k=args.top_k
    )


if __name__ == "__main__":
    main()
