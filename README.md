# CLIP ONNX 语言检测

## Environment Setup

```bash

# create a virtual environment
conda create -n clip_lang python=3.11

# activate the virtual environment
conda activate clip_lang

# install dependencies
pip install -r requirements.txt
```

## 使用方法

### 1. 导出模型

```bash
# 使用镜像加速（可选）
export HF_ENDPOINT=https://hf-mirror.com

# 导出所有预定义模型
python export_models.py

```

导出的模型文件将保存在当前目录下：

- `export_models/clip_tiny.onnx`
- `export_models/clip_small.onnx`
- `export_models/clip_large.onnx`

### 2. 推理

#### 命令行方式（推荐）

```bash
# 对单个图像文件推理
python inference.py --model onnx_models/clip_large.onnx --images ./test_files/jp_hand.jpg

# 对多个图像文件推理
python inference.py --model onnx_models/clip_small.onnx --images image1.jpg image2.jpg image3.jpg

# 对整个目录的图像文件推理
python inference.py --model onnx_models/clip_large.onnx --images ./test_files/

# 显示前3个最可能的语言结果
python inference.py --model onnx_models/clip_large.onnx --images ./test_files/ --top-k 3
```
