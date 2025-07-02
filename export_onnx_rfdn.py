import torch
from RFDN import RFDN  # 导入模型结构

# === 加载模型 ===
model = RFDN(upscale=4)
ckpt = torch.load('trained_model/aim_rfdn.pth', map_location='cpu')
model.load_state_dict(ckpt['params'] if 'params' in ckpt else ckpt)
model.eval()

# === 准备输入张量（动态尺寸支持）===
dummy_input = torch.randn(1, 3, 128, 128)

# === 导出为 ONNX 文件 ===
torch.onnx.export(
    model,
    dummy_input,
    'rfdn_x4.onnx',
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {2: "height", 3: "width"},
        "output": {2: "height_out", 3: "width_out"}
    }
)

print("✅ 导出成功：rfdn_x4.onnx")