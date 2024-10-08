import torch
import argparse
import yaml
import math

from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
from semseg.metrics import *
from multi_pert import PerturbModelTwo

from rich.console import Console

console = Console()


class SemSeg:
    def __init__(self, cfg, pert_ratio=0, target_only_positive=True) -> None:
        self.pert_ratio = pert_ratio
        self.person_idx = 12
        self.car_idx = 20
        self.tree_idx = 4

        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES
        self.classes = len(self.labels)

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette))
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))

        """
            多类别加噪， 人大，车or树小
        """
        # 定义目标类别的扰动设置
        target_perturbations = [
            {'index': self.person_idx, 'positive': True} # 对人类别添加正数噪声
            # {'index': self.car_idx, 'positive': False},
            # {'index': self.tree_idx, 'positive': False} # 对车or树类别添加负数噪声
        ]

        # 初始化扰动模型
        perturb_model = PerturbModelTwo(
            model=self.model,
            pert_ratio=self.pert_ratio,
            target_perturbations=target_perturbations
        )

        self.model = perturb_model
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])


    def preprocess(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]
        console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H * scale_factor), round(W * scale_factor)

        if image.shape[0] == 4:
            image = image[:3, :, :]  # 保留RGB通道，去掉Alpha通道
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        # resize the image
        image = T.Resize((nH, nW))(image)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Image.Image:
        # 调整到原始图像大小
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)

        # 获取预测的类别索引
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int).squeeze(0)

        # 定义颜色
        yellow = torch.tensor([255, 255, 0], dtype=torch.uint8)  # 人
        blue = torch.tensor([0, 0, 255], dtype=torch.uint8)  # 车
        green = torch.tensor([0, 255, 0], dtype=torch.uint8)  # 树
        purple = torch.tensor([128, 0, 128], dtype=torch.uint8)  # 其他类别

        # 初始化一个空的颜色映射图
        seg_image = torch.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=torch.uint8)

        # 应用颜色映射
        seg_image[seg_map == self.person_idx] = yellow  # 人
        seg_image[seg_map == self.car_idx] = blue  # 车
        seg_image[seg_map == self.tree_idx] = green  # 树
        # seg_image[(seg_map != self.person_idx) &
        #           (seg_map != self.car_idx) &
        #           (seg_map != self.tree_idx)] = purple # 其他类别
        # seg_image[(seg_map != self.person_idx) &
        #           (seg_map != self.tree_idx)] = purple  # 其他类别

        if overlay:
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        # 如果原图有4个通道 (RGBA)，去掉Alpha通道
        if orig_img.shape[0] == 4:
            orig_img = orig_img[:3, :, :]  # 只保留前三个通道(RGB)

        if overlay:
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        # 转换为 uint8 类型
        seg_image = seg_image.byte()

        # 转换为 PIL 图像
        seg_image_pil = Image.fromarray(seg_image.numpy())

        return seg_image_pil

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)

    def predict(self, img_fname: str, overlay: bool) -> Image.Image:
        image = io.read_image(img_fname)
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Dataset > [red]{cfg['DATASET']['NAME']}[/red]")

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(parents=True, exist_ok=True)

    # 定义一组扰动比例
    pert_ratio_list = [round(0.2 * i, 2) for i in range(21)]

    with console.status("[bright_green]Processing..."):
        for idx, pert_ratio in enumerate(pert_ratio_list):
            console.rule(f"[yellow]Perturbation Ratio: {pert_ratio}[/yellow]")
            semseg = SemSeg(cfg, pert_ratio=pert_ratio)

            if test_file.is_file():
                console.rule(f'[green]{test_file.name}')
                segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
                # 创建对应的子文件夹以保存不同扰动比例的结果
                ratio_dir = save_dir / f"pert_ratio_{pert_ratio}"
                ratio_dir.mkdir(parents=True, exist_ok=True)
                segmap.save(ratio_dir / f"{test_file.stem}.png")
            else:
                files = test_file.glob('*.*')
                for file in files:
                    console.rule(f'[green]{file.name}')
                    segmap = semseg.predict(str(file), cfg['TEST']['OVERLAY'])
                    # 创建对应的子文件夹以保存不同扰动比例的结果
                    ratio_dir = save_dir / f"pert_ratio_{pert_ratio}"
                    ratio_dir.mkdir(parents=True, exist_ok=True)
                    segmap.save(ratio_dir / f"{file.stem}.png")

    console.rule(f"[cyan]Segmentation results are saved in `{save_dir}`")