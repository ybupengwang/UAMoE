import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from config import Config
from model import Farnet
from thop import profile
import time
def load_model(model, checkpoint_path, device='cuda:0'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model = model.to(device)
    model.eval()
    return model

def predict(model, img_path, device='cuda:0'):
    # 加载图像
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size  # (宽, 高)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((Config.resize_h, Config.resize_w)),
        transforms.ToTensor()
    ])
    img_data = transform(img).unsqueeze(0).to(device)  # 添加 batch 维度，变成 (1, C, H, W)

    # 模型前向推理
    with torch.no_grad():
        outputs = model(img_data)
        # 这里根据你的模型输出调整解包逻辑
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 5:
            refined_coords = outputs[4]
        else:
            raise ValueError("模型输出格式不正确，无法获取 refined_coords")

    # 坐标还原到原始图像尺寸
    refined_coords = refined_coords.cpu().numpy()  # (B, N, 2)
    refined_coords[:, :, 0] = refined_coords[:, :, 0] * (img_w - 1)
    refined_coords[:, :, 1] = refined_coords[:, :, 1] * (img_h - 1)

    return refined_coords[0]  # 返回第一个样本的坐标


#计算参数量和FLOPs
model = Farnet().cuda()
model.eval()

# ⚠️ 用一张标准输入尺寸（必须和你训练一致）
dummy_input = torch.randn(1, 3, 512, 512).cuda()

flops, params = profile(model, inputs=(dummy_input,))

print("FLOPs: {:.4f}G".format(flops*2 / 1e9))
print("Params: {:.4f}M".format(params / 1e6))

model = model.to("cuda")
dummy_input = dummy_input.to("cuda")

torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    for _ in range(50):
        _ = model(dummy_input)

torch.cuda.synchronize()
end = time.time()

gpu_time = (end - start) / 50 * 1000
print("GPU Inference Time: {:.3f} ms".format(gpu_time))

model_cpu = model.to("cpu")
dummy_cpu = torch.randn(1, 3, 512, 512)

start = time.time()

with torch.no_grad():
    for _ in range(50):
        _ = model_cpu(dummy_cpu)

end = time.time()

cpu_time = (end - start) / 50 * 1000
print("CPU Inference Time: {:.3f} ms".format(cpu_time))





# # 模型初始化（你需要根据自己模型定义来构建）
# model = Farnet()
# checkpoint_path = "best_mode_youHKDceshi.pth"
# model = load_model(model, checkpoint_path, device='cuda:0')
# # 预测图像关键点
# # img_path = r"D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\jingzhuitrainimg\11810468.jpg"
# with open("keshihua/test_list.txt", 'r') as f:
#     img_paths = [line.strip() for line in f.readlines()]
#
# for img_path in img_paths:
#     keypoints = predict(model, img_path, device='cuda:0')
#     print(keypoints)  # (N, 2)，表示 N 个关键点





