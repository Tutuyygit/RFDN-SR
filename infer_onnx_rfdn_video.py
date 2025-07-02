import onnxruntime as ort
import cv2
import numpy as np
import os
import time

# === 设置路径 ===
onnx_path = "rfdn_x4.onnx"
input_video = "input_720p.mp4"
output_video = "output_4k_onnx.mp4"

# === 初始化 ONNX 模型 ===
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# === 打开输入视频 ===
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === 设置输出视频参数（4x 放大）===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width*4, height*4))

frame_count = 0
total_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === 计时开始 ===
    start = time.time()

    # === 图像预处理 ===
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

    # === ONNX 推理 ===
    out_img = session.run([output_name], {input_name: img})[0]
    out_img = np.clip(out_img.squeeze(), 0, 1)

    # === 后处理 + 保存 ===
    out_img = (out_img * 255.0).astype(np.uint8)
    out_img = np.transpose(out_img, (1, 2, 0))  # (H, W, 3)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    out.write(out_img)

    frame_count += 1
    elapsed = time.time() - start
    total_time += elapsed
    print(f"帧 {frame_count} 用时: {elapsed:.3f}s")

cap.release()
out.release()
print(f"\n✅ 完成超分视频推理！平均每帧 {total_time/frame_count:.3f}s，输出保存为：{os.path.abspath(output_video)}")