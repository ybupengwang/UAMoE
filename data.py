import os
from config import Config
#import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from utils import get_markposion_fromtxt
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import gaussian_blur
import random
Config = Config()
from scipy.ndimage import gaussian_filter
from itertools import product
from typing import Tuple, Optional
class medical_dataset(Dataset):
    def __init__(self, img_dir, gt_dir, resize_height, resize_width, point_num, sigma, transform=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.img_names = os.listdir(img_dir)
        self.img_nums = len(self.img_names)
        self.point_num = point_num
        self.sigma = sigma
        self.heatmap_height = int(self.resize_height)
        self.heatmap_width = int(self.resize_width)
        self.transform = transform

    def __getitem__(self, i):
        #index = i % self.img_nums
        img_name = self.img_names[i]
        img_path = os.path.join(self.img_dir, img_name)
        img, img_w,img_h,scal_ratio_w, scal_ratio_h = self.img_preproccess(img_path)
        # img = normalize_robust(img)
        gt_path = self.gt_dir + '/' + img_name.split('.')[0] + '.txt'
        gt_x, gt_y = get_markposion_fromtxt(self.point_num, gt_path)
        x_all = gt_x / scal_ratio_w   #x_all 为缩放后的坐标
        y_all = gt_y / scal_ratio_h

        if self.transform:
            img,x_all,y_all,gt_x,gt_y = self.data_augmentation(img,x_all,y_all,gt_x,gt_y)

        # # === 添加坐标扰动 ===
        #     delta = 10  # 设置扰动阈值
        #     lambda_x = np.random.uniform(-delta, delta, size=x_all.shape)
        #     lambda_y = np.random.uniform(-delta, delta, size=y_all.shape)
        #
        #     x_all = np.clip(x_all + lambda_x, 0, self.resize_width)  # 约束在图像范围内
        #     y_all = np.clip(y_all + lambda_y, 0, self.resize_height)

        #计算热图
        heatmaps = self.get_heatmaps(x_all, y_all, self.sigma)
        #heatmaps = self.generate_udp_gaussian_heatmaps(heatmap_size=(self.resize_width,self.resize_height),keypoints=(x_all[i],y_all[i]),sigma = self.sigma)
        heatmaps_refine = self.get_refine_heatmaps(x_all / 2, y_all / 2, self.sigma)
        heatmaps_hrnet = self.heatmaps_hrnet(x_all / 4, y_all / 4, self.sigma)

        # img = self.data_preproccess(img)
        heatmaps = self.data_preproccess(heatmaps)
        heatmaps_refine = self.data_preproccess(heatmaps_refine)

        heatmaps_hrnet = self.data_preproccess(heatmaps_hrnet)

        #测试热图DAPK 属于后处理
        # batch_heatmaps = heatmaps.detach().cpu().numpy()
        # coords= get_final_preds(batch_heatmaps[np.newaxis, :, :, :])




        return img, img_w,img_h,heatmaps, heatmaps_refine, img_name, x_all, y_all,gt_x,gt_y,heatmaps_hrnet

    def data_augmentation(self, img, x_all, y_all,gt_x,gt_y):
        """数据增强，应用到图像和关键点"""
        _, h, w = img.shape  # 获取图像尺寸 (C, H, W)
        # 转换为地标格式 [(x1, y1), (x2, y2), ...]
        landmarks = list(zip(x_all.tolist(), y_all.tolist()))
        # 随机水平翻转
        # if random.random() > 0.5:
        #     img = TF.hflip(img)
        #     x_all = (w-1) - x_all  # 水平翻转关键点
        #     gt_x = (Config.resize_w*Config.scal_w-1) -gt_x
        #     landmarks = [(w-1 - x, y) for (x, y) in landmarks]

        # 随机垂直翻转
        # if random.random() > 0.5:
        #     img = TF.vflip(img)
        #     y_all = (h-1) - y_all  # 垂直翻转关键点
        #     gt_y = (Config.resize_h*Config.scal_h-1) - gt_y
        #     landmarks = [(x, h-1 - y) for (x, y) in landmarks]

        # 随机旋转（-30° ~ 30°）
        # angle = random.uniform(-10, 10)
        # img = TF.rotate(img, angle)
        # x_all, y_all = self.rotate_points(x_all, y_all, angle, w, h)
        # gt_x, gt_y = self.rotate_points(gt_x, gt_y, angle, Config.resize_w*Config.scal_w, Config.resize_h*Config.scal_h)
        # 随机颜色抖动
        # img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
        # img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))

        # if random.random() > 0.2:  # 控制是否应用该增强
        #     img = self.perturb_region_proposal(img, landmarks)

        return img, x_all, y_all,gt_x,gt_y

    def rotate_points(self, x_all, y_all, angle, width, height):
        """对关键点进行旋转变换"""
        angle = np.deg2rad(angle)
        center_x, center_y = (width-1) / 2, (height-1) / 2  # 旋转中心
        x_new = (x_all - center_x) * np.cos(angle) - (y_all - center_y) * np.sin(angle) + center_x
        y_new = (x_all - center_x) * np.sin(angle) + (y_all - center_y) * np.cos(angle) + center_y
        return x_new, y_new

    def perturb_region_proposal(
            self,
            image: torch.Tensor,
            landmarks: list,
            min_R: int = 20,
            max_R: int = 60,
            k: int = 3,
            blur_size: int = 15
    ) -> torch.Tensor:
        """
        基于地标扰动的模糊区域增强。
        """
        C, H, W = image.shape
        image_blurred = image.clone()

        if len(landmarks) < k:
            k = len(landmarks)

        selected_landmarks = random.sample(landmarks, 17)

        for (x_mu, y_mu) in selected_landmarks:
            x_sigma, y_sigma = 5, 5
            x = int(np.random.normal(x_mu, x_sigma))
            y = int(np.random.normal(y_mu, y_sigma))

            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))

            h = random.randint(min_R, max_R)
            w = random.randint(min_R, max_R)

            top = max(0, y - h // 2)
            bottom = min(H, y + h // 2)
            left = max(0, x - w // 2)
            right = min(W, x + w // 2)

            region = image[:, top:bottom, left:right]
            if region.shape[1] > 0 and region.shape[2] > 0:
                blurred_region = gaussian_blur(region, kernel_size=[blur_size, blur_size])
                image_blurred[:, top:bottom, left:right] = blurred_region

        return image_blurred

    def __len__(self):
        return self.img_nums

    def get_heatmaps(self, x_all, y_all, sigma):
        # heatmaps = np.zeros((self.point_num, self.heatmap_height, self.heatmap_width))
        # for i in range(self.point_num):
        #     #heatmaps[i] = CenterLabelHeatMap(self.heatmap_width, self.heatmap_height, x_all[i], y_all[i], sigma)
        #
        #     heatmaps[i] = generate_udp_gaussian_heatmaps(heatmap_size = (self.resize_width, self.resize_height), keypoints = keypoints, sigma = self.sigma)
        # heatmaps = np.asarray(heatmaps, dtype="float32")
        keypoints = np.stack([x_all, y_all], axis=1)[np.newaxis, ...].astype(np.float32)
        heatmaps = generate_udp_gaussian_heatmaps(heatmap_size=(self.resize_width, self.resize_height),keypoints=keypoints, sigma=self.sigma)

        return heatmaps

    def get_refine_heatmaps(self, x_all, y_all, sigma):
        heatmaps = np.zeros((self.point_num, int(self.heatmap_height / 2), int(self.heatmap_width / 2)))
        for i in range(self.point_num):
            heatmaps[i] = CenterLabelHeatMap(int(self.heatmap_width / 2), int(self.heatmap_height / 2), x_all[i],y_all[i], sigma)

        heatmaps = np.asarray(heatmaps, dtype="float32")
        return heatmaps
    def heatmaps_hrnet(self, x_all, y_all, sigma):
        heatmaps = np.zeros((self.point_num, int(self.heatmap_height / 4), int(self.heatmap_width / 4)))
        for i in range(self.point_num):
            heatmaps[i] = CenterLabelHeatMap(int(self.heatmap_width / 4), int(self.heatmap_height / 4), x_all[i],
                                             y_all[i], sigma)
        heatmaps = np.asarray(heatmaps, dtype="float32")
        return heatmaps



    def img_preproccess(self, img_path):
        img = Image.open(img_path)
        img_w, img_h = img.size
        transform = transforms.Compose([
            transforms.Resize((Config.resize_h, Config.resize_w)),  # 调整大小
            transforms.ToTensor(),  # 转换为tensor，已经将其转为 (C, H, W) 格式
        ])
        img = transform(img)
        if img.shape[0]==1:
            img = img.repeat(3, 1, 1)
        scal_ratio_w = (img_w-1) / (self.resize_width-1)
        scal_ratio_h = (img_h-1) / (self.resize_height-1)
        transform1 = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        )
        img = transform1(img)

        return img, img_w,img_h,scal_ratio_w, scal_ratio_h

    def data_preproccess(self, data):
        data = torch.from_numpy(data).float()
        return data

def adaptive_sigma(epoch, sigma_0=10, decay=0.01):
    """根据 epoch 计算 sigma"""
    return sigma_0 * np.exp(-decay * epoch)



def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma, radius=50):
    x = np.arange(0, img_width, dtype=np.float32)
    y = np.arange(0, img_height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    heatmap = np.exp(-((xx - c_x) ** 2 + (yy - c_y) ** 2) / (2 * sigma ** 2))
    heatmap[heatmap < np.finfo(np.float32).eps * heatmap.max()] = 0  # 稀疏优化（可选）

    return heatmap

def generate_udp_gaussian_heatmaps(
        heatmap_size: Tuple[int, int],
        keypoints: np.ndarray,

        sigma: float,
) -> Tuple[np.ndarray]:
    """Generate gaussian heatmaps of keypoints using `UDP`_.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        sigma (float): The sigma value of the Gaussian heatmap

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape
            (K, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size
    keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    # 3-sigma rule
    radius = sigma * 3

    # xy grid
    gaussian_size = 2 * radius + 1
    x = np.arange(0, gaussian_size, 1, dtype=np.float32)
    y = x[:, None]

    for n, k in product(range(N), range(K)):
        # skip unlabled keypoints
        if keypoints_visible[n, k] < 0.5:
            continue

        mu = (keypoints[n, k] + 0.5).astype(np.int64)
        # check that the gaussian has in-bounds part
        left, top = (mu - radius).astype(np.int64)
        right, bottom = (mu + radius + 1).astype(np.int64)

        if left >= W or top >= H or right < 0 or bottom < 0:
            keypoint_weights[n, k] = 0
            continue

        mu_ac = keypoints[n, k]
        x0 = y0 = gaussian_size // 2
        x0 += mu_ac[0] - mu[0]
        y0 += mu_ac[1] - mu[1]
        gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # valid range in gaussian
        g_x1 = max(0, -left)
        g_x2 = min(W, right) - left
        g_y1 = max(0, -top)
        g_y2 = min(H, bottom) - top

        # valid range in heatmap
        h_x1 = max(0, left)
        h_x2 = min(W, right)
        h_y1 = max(0, top)
        h_y2 = min(H, bottom)

        heatmap_region = heatmaps[k, h_y1:h_y2, h_x1:h_x2]
        gaussian_regsion = gaussian[g_y1:g_y2, g_x1:g_x2]

        _ = np.maximum(heatmap_region, gaussian_regsion, out=heatmap_region)

    return heatmaps

#获取最大预测
def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''

    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

#泰勒展开
def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
            + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.matrix([[dx],[dy]])
        hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord




def gaussian_blur1(hm, kernel):
    sigma = kernel / 6  # 经验公式：kernel=6*sigma 可近似匹配 OpenCV 效果
    border = (kernel - 1) // 2
    batch_size, num_joints, height, width = hm.shape

    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(hm[i, j])
            # 创建带边界的扩展热图
            dr = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
            dr[border: -border, border: -border] = hm[i, j].copy()
            # 使用 scipy 进行高斯模糊
            dr = gaussian_filter(dr, sigma=sigma)
            # 裁剪回原始大小
            cropped = dr[border: -border, border: -border]
            if np.max(cropped) > 0:
                hm[i, j] = cropped * (origin_max / np.max(cropped))
            else:
                hm[i, j] = cropped
    return hm
def get_final_preds( hm, center=None, scale=None):

    coords, maxvals = get_max_preds(hm)
    heatmap_height = hm.shape[2]
    heatmap_width = hm.shape[3]

    # post-processing
    hm = gaussian_blur1(hm, 11)
    hm = np.maximum(hm, 1e-10)
    hm = np.log(hm)
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            coords[n,p] = taylor(hm[n][p], coords[n][p])

    preds = coords.copy()

    # # Transform back
    # for i in range(coords.shape[0]):
    #     preds[i] = transform_preds(
    #         coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
    #     )

    return preds,maxvals



def refine_keypoints_dark_udp(keypoints: np.ndarray, heatmaps: np.ndarray,
                              blur_kernel_size: int) -> np.ndarray:
    """Refine keypoint predictions using distribution aware coordinate decoding
    for UDP. See `UDP`_ for details. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50., heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(
        heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum('imn,ink->imk', hessian,
                                  derivative).squeeze()

    return keypoints