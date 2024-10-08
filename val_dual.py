import torch
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from tools.double_side_pert import PerturbModelDual

import csv
from torch.cuda.amp import autocast
import os

@torch.no_grad()
def evaluate(model, dataloader, device, perturb_model=None):
    print('Evaluating...')
    model.eval()
    if perturb_model is not None:
        perturb_model.eval()

    metrics = Metrics(dataloader.dataset.n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        with autocast():
            if perturb_model is not None:
                logits = perturb_model(images)
            else:
                logits = model(images)

            preds = logits.softmax(dim=1)
        metrics.update(preds, labels)

    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()

    return acc, macc, f1, mf1, ious, miou

def save_results_to_txt(results, file_path):
    with open(file_path, 'a') as f:
        for result in results:
            f.write(result + '\n')

def save_results_to_csv(row, file_path):
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)  # num_workers 设置为1以减少潜在开销
    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists():
        model_path = Path(
            cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
    print(f"Evaluating {model_path}...")

    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model = model.to(device)

    # 不在此处实例化扰动模型，而是在需要时进行实例化
    perturb_model = None

    # 获取所有类别名称
    class_names = dataset.CLASSES
    n_classes = dataset.n_classes

    # 定义扰动比例（0.0 到 20.0，步长 0.2）
    pert_ratio_list = [round(x * 0.2, 1) for x in range(0, 101)]  # 0.0 到 20.0，步长 0.2

    # 初始化 CSV 文件的表头
    csv_headers = ['Perturb Ratio'] + \
                  [f'IoU_{name}' for name in class_names] + ['Mean IoU'] + \
                  [f'F1_{name}' for name in class_names] + ['Mean F1'] + \
                  [f'Acc_{name}' for name in class_names] + ['Mean Acc']

    # 遍历每个扰动比例
    for pert_ratio in pert_ratio_list:
        # 创建对应的文件夹
        ratio_folder = Path(cfg['SAVE_DIR']) / f"pert_ratio_{pert_ratio}"
        ratio_folder.mkdir(parents=True, exist_ok=True)

        # 定义保存文件路径
        txt_file_path = ratio_folder / 'evaluation_results.txt'
        csv_file_path = ratio_folder / 'evaluation_results.csv'

        # 检查是否已经存在结果文件，若存在则跳过该扰动比例
        if txt_file_path.exists() and csv_file_path.exists():
            print(f"Results for perturb ratio {pert_ratio} already exist. Skipping...")
            continue

        # 设置当前扰动比例
        if pert_ratio == 0.0:
            current_perturb_model = None  # 使用原始模型
            print(f"\nEvaluating with perturb ratio: {pert_ratio} (Using Original Model)")
        else:
            # 实例化扰动模型（仅实例化一次）
            if perturb_model is None:
                perturb_model = PerturbModelDual(model=model, pert_ratio=pert_ratio).to(device)
            else:
                perturb_model.set_pert_ratio(pert_ratio)
            current_perturb_model = perturb_model
            print(f"\nEvaluating with perturb ratio: {pert_ratio} (Using PerturbModelDual)")

        # 进行评估
        if eval_cfg['MSF']['ENABLE']:
            acc, macc, f1, mf1, ious, miou = evaluate_msf(
                model, dataloader, device,
                eval_cfg['MSF']['SCALES'],
                eval_cfg['MSF']['FLIP'],
                current_perturb_model
            )
        else:
            acc, macc, f1, mf1, ious, miou = evaluate(
                model, dataloader, device, current_perturb_model
            )

        # 准备TXT格式的结果
        result_txt = f"Perturb Ratio: {pert_ratio}\n"
        for idx, class_name in enumerate(class_names):
            result_txt += f"  {class_name} - IoU: {ious[idx]:.4f}, F1: {f1[idx]:.4f}, Acc: {acc[idx]:.4f}\n"
        result_txt += f"  Mean IoU: {miou:.4f}, Mean F1: {mf1:.4f}, Mean Acc: {macc:.4f}\n"

        # 保存TXT结果到文件
        save_results_to_txt([result_txt], txt_file_path)

        # 准备CSV格式的结果
        csv_row = [pert_ratio] + \
                  [f"{ious[idx]:.4f}" for idx in range(n_classes)] + [f"{miou:.4f}"] + \
                  [f"{f1[idx]:.4f}" for idx in range(n_classes)] + [f"{mf1:.4f}"] + \
                  [f"{acc[idx]:.4f}" for idx in range(n_classes)] + [f"{macc:.4f}"]

        # 保存CSV结果到文件
        save_results_to_csv(csv_row, csv_file_path)

        # **在控制台打印当前pert_ratio的结果**
        print(result_txt)

    # 最终结果保存提示
    print(f"All results saved to {cfg['SAVE_DIR']}")

def evaluate_msf(model, dataloader, device, scales, flip, perturb_model=None):
    model.eval()
    if perturb_model is not None:
        perturb_model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True)
            scaled_images = scaled_images.to(device)

            with autocast():
                if perturb_model is not None:
                    logits = perturb_model(scaled_images)
                else:
                    logits = model(scaled_images)

                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

                if flip:
                    scaled_images_flipped = torch.flip(scaled_images, dims=(3,))
                    if perturb_model is not None:
                        logits_flipped = perturb_model(scaled_images_flipped)
                    else:
                        logits_flipped = model(scaled_images_flipped)
                    logits_flipped = torch.flip(logits_flipped, dims=(3,))
                    logits_flipped = F.interpolate(logits_flipped, size=(H, W), mode='bilinear', align_corners=True)
                    scaled_logits += logits_flipped.softmax(dim=1)

        metrics.update(scaled_logits, labels)

    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)
