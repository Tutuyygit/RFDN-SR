import os
import cv2
import torch
import numpy as np
import time
from RFDN import RFDN  # 导入模型

# === 1. 加载模型 ===
print("🚀 正在加载 RFDN 模型 ...")
model = RFDN()
model_path = 'trained_model/aim_rfdn.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# === 2. 打开输入视频 ===
video_path = 'input_720p.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"❌ 无法打开视频：{video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"🎥 视频信息: {width}x{height} @ {fps:.0f}fps")

# === 3. 创建输出视频写入器 ===
out_path = 'output_4k.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width * 4, height * 4))

frame_idx = 0
total_start = time.time()

# === 4. 帧循环 ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame_rgb.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        output = model(img).clamp(0, 1).cpu().numpy()

    output_img = np.transpose(output[0], (1, 2, 0)) * 255.0
    output_img = output_img.round().astype(np.uint8)
    output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    out_writer.write(output_bgr)

    end = time.time()
    print(f"✅ 帧 {frame_idx} 完成，用时 {end - start:.2f} 秒")
    frame_idx += 1

# === 5. 清理资源 ===
cap.release()
out_writer.release()
total_time = time.time() - total_start
print(f"🎞️ 处理完成，共 {frame_idx} 帧，总耗时 {total_time:.2f} 秒")
print(f"📁 输出视频路径：{os.path.abspath(out_path)}")