import torch
import cv2
import numpy as np
from RFDN import RFDN  # 确保 RFDN.py 在同目录下

# === 1. 加载模型 ===
print("🚀 正在加载 RFDN 模型 ...")
model = RFDN()  # 默认结构，不传参数
model_path = 'trained_model/aim_rfdn.pth'  # 替换为你的模型文件名
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# === 2. 加载输入图像 ===
img = cv2.imread("input.png")
if img is None:
    raise Exception("❌ 无法读取 input.png，请确认文件存在")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # HWC -> CHW -> NCHW

# === 3. 推理（CPU）
print("🔍 开始图像超分推理 ...")
with torch.no_grad():
    output = model(img).clamp(0, 1).cpu().squeeze(0).numpy()

# === 4. 保存输出图像 ===
output_img = (output.transpose(1, 2, 0) * 255).round().astype(np.uint8)
output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_4k.png", output_img)
print("✅ 超分完成，输出已保存为 output_4k.png")