import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import adaptive_sigma
from config import Config
from data import medical_dataset
from model import UAMoE
from test import get_errors
from train import train_model
from samplingArgmax import SamplingArgmax
from loss import JointsOHKMMSELoss
from soft_argmax import SoftArgmax
from utils import setup_seed
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
if __name__ == '__main__':
    setup_seed(2025)
    model = UAMoE()
    model.cuda(Config.GPU)

    # hrnet_path = "HR/pose_hrnet_w48_384x288.pth"
    # state_dict = torch.load(hrnet_path, map_location=torch.device(Config.GPU))
    # hrnet_dict = model.backbone.state_dict()
    # filtered_state_dict = {k: v for k, v in state_dict.items() if k in hrnet_dict and v.shape == hrnet_dict[k].shape}
    # hrnet_dict.update(filtered_state_dict)

    # model_weights_path = "best_mode_wuHKD.pth"  # 替换成你的权重路径
    # state_dict = torch.load(model_weights_path, map_location=torch.device(Config.GPU))
    # model_dict = model.state_dict()
    # filtered_state_dict = {k: v for k, v in state_dict.items()
    #                        if k in model_dict and v.shape == model_dict[k].shape}
    # model_dict.update(filtered_state_dict)
    #
    # model.load_state_dict(model_dict)
    # for param in model.parameters():
    #     param.requires_grad = True
    # for name, param in model.named_parameters():
    #     if "structure" in name or "moe" in name:
    #         param.requires_grad = True
    #
    # print([(name, param.requires_grad) for name, param in model.named_parameters()])

    soft_argmax = SoftArgmax()

    # head 数据集
    test_set1 = medical_dataset(Config.test_img_dir1, Config.test_gt_dir1, Config.resize_h, Config.resize_w,Config.point_num, Config.sigma, transform=False)
    test_loader = DataLoader(dataset=test_set1, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,prefetch_factor=2, persistent_workers=True)
    train_set = medical_dataset(Config.img_dir, Config.gt_dir, Config.resize_h, Config.resize_w, Config.point_num,sigma=Config.sigma, transform=True)
    train_loader = DataLoader(dataset=train_set, batch_size=3, shuffle=True, num_workers=4,pin_memory=True,prefetch_factor=2,persistent_workers=True)
    #手数据
    # test_set1 = medical_dataset(Config.testhandimg_dir, Config.testhandgt_dir, Config.resize_h, Config.resize_w,Config.point_num, Config.sigma, transform=False)
    # test_loader = DataLoader(dataset=test_set1, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,prefetch_factor=2, persistent_workers=True)
    # train_set = medical_dataset(Config.handimg_dir, Config.handgt_dir, Config.resize_h, Config.resize_w, Config.point_num,sigma=Config.sigma, transform=True)
    # train_loader = DataLoader(dataset=train_set, batch_size=3, shuffle=True, num_workers=4,pin_memory=True,prefetch_factor=2,persistent_workers=True)

    criterion = JointsOHKMMSELoss(use_target_weight=False).cuda(Config.GPU)
    #optimizer_ft = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-5)

    if Config.optimizer == 'adamW':
        optimizer_ft = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-5)
    elif Config.optimizer == 'adam':
        optimizer_ft = torch.optim.Adam(model.parameters(), lr=Config.lr)
    elif Config.optimizer == 'sgd':
        optimizer_ft = torch.optim.SGD(model.parameters(), lr=Config.lr, momentum=0.9, weight_decay=1e-4)
    elif Config.optimizer == 'rmsprop':
        optimizer_ft = torch.optim.RMSprop(model.parameters(), lr=Config.lr, alpha=0.99)
    elif Config.optimizer == 'adagrad':
        optimizer_ft = torch.optim.Adagrad(model.parameters(), lr=Config.lr)

    scheduler = CosineAnnealingLR(optimizer_ft, T_max=Config.num_epochs)#余弦退火

    for epoch in range(Config.num_epochs):
        if epoch%300 ==0:
            sigma = adaptive_sigma(epoch)
        #print("sigma："+str(sigma))
        for param_group in optimizer_ft.param_groups:
            print(f"sigma： {sigma}, Learning Rate: {param_group['lr']}")
        train_model(model, soft_argmax,criterion, optimizer_ft, scheduler, train_loader, test_loader, Config.num_epochs, epoch)



