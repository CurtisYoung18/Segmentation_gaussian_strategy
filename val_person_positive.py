import torch
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from torch.cuda.amp import autocast

import os

# Import the modified PerturbModelDual
from one_target_pert import PerturbModelOneTarget

def save_results_to_txt(results, file_path):
    with open(file_path, 'a') as f:
        for result in results:
            f.write(result + '\n')

@torch.no_grad()
def evaluate(model, dataloader, device, perturb_model=None, person_class_idx=None):
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

    # Extract metrics for 'person' class
    if person_class_idx is not None:
        person_iou = ious[person_class_idx]
        person_acc = acc[person_class_idx]
    else:
        person_iou = None
        person_acc = None

    return acc, macc, f1, mf1, ious, miou, person_acc, person_iou

def evaluate_msf(model, dataloader, device, scales, flip, perturb_model=None, person_class_idx=None):
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

    # Extract metrics for 'person' class
    if person_class_idx is not None:
        person_iou = ious[person_class_idx]
        person_acc = acc[person_class_idx]
    else:
        person_iou = None
        person_acc = None

    return acc, macc, f1, mf1, ious, miou, person_acc, person_iou

def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)
    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists():
        model_path = Path(
            cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
    print(f"Evaluating {model_path}...")

    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model = model.to(device)

    perturb_model = None

    class_names = dataset.CLASSES
    n_classes = dataset.n_classes

    pert_ratio_list = [round(x * 0.2, 1) for x in range(0, 101)]  # 0.0 to 20.0, step 0.2

    # Get index of 'person' class
    person_class_idx = None
    for idx, name in enumerate(class_names):
        if name.lower() == 'person':
            person_class_idx = idx
            break

    if person_class_idx is None:
        raise ValueError("Class 'person' not found in class names.")

    # Initialize metrics summary file
    metrics_file = Path(cfg['SAVE_DIR']) / 'metrics_summary.txt'
    with open(metrics_file, 'w') as f:
        f.write("pert_ratios = [\n")
        f.write("]\n\n")
        f.write("mean_ious = [\n")
        f.write("]\n\n")
        f.write("mean_f1s = [\n")
        f.write("]\n\n")
        f.write("mean_accs = [\n")
        f.write("]\n\n")
        f.write("person_ious = [\n")
        f.write("]\n\n")
        f.write("person_accs = [\n")
        f.write("]\n")

    for pert_ratio in pert_ratio_list:
        ratio_folder = Path(cfg['SAVE_DIR']) / f"pert_ratio_{pert_ratio}"
        ratio_folder.mkdir(parents=True, exist_ok=True)

        txt_file_path = ratio_folder / 'evaluation_results.txt'

        if txt_file_path.exists():
            print(f"Results for perturb ratio {pert_ratio} already exist. Skipping...")
            continue

        if pert_ratio == 0.0:
            current_perturb_model = None  # Use original model
            print(f"\nEvaluating with perturb ratio: {pert_ratio} (Using Original Model)")
        else:
            if perturb_model is None:
                perturb_model = PerturbModelOneTarget(model=model, pert_ratio=pert_ratio, target_class_idx=person_class_idx).to(device)
            else:
                perturb_model.set_pert_ratio(pert_ratio)
            current_perturb_model = perturb_model
            print(f"\nEvaluating with perturb ratio: {pert_ratio} (Using PerturbModelDual)")

        if eval_cfg['MSF']['ENABLE']:
            acc, macc, f1, mf1, ious, miou, person_acc, person_iou = evaluate_msf(
                model, dataloader, device,
                eval_cfg['MSF']['SCALES'],
                eval_cfg['MSF']['FLIP'],
                current_perturb_model,
                person_class_idx
            )
        else:
            acc, macc, f1, mf1, ious, miou, person_acc, person_iou = evaluate(
                model, dataloader, device, current_perturb_model, person_class_idx
            )

        # Prepare TXT results
        result_txt = f"Perturb Ratio: {pert_ratio}\n"
        for idx, class_name in enumerate(class_names):
            result_txt += f"  {class_name} - IoU: {ious[idx]:.4f}, F1: {f1[idx]:.4f}, Acc: {acc[idx]:.4f}\n"
        result_txt += f"  Mean IoU: {miou:.4f}, Mean F1: {mf1:.4f}, Mean Acc: {macc:.4f}\n"
        result_txt += f"  Person IoU: {person_iou:.4f}, Person Acc: {person_acc:.4f}\n"

        # Save TXT results to file
        save_results_to_txt([result_txt], txt_file_path)

        # Print results to console
        print(result_txt)

        # Append metrics to summary file
        with open(metrics_file, 'r') as f:
            lines = f.readlines()

        # Update pert_ratios
        idx = lines.index("pert_ratios = [\n") + 1
        lines[idx] = lines[idx].strip() + f"{pert_ratio},\n"

        # Update mean_ious
        idx = lines.index("mean_ious = [\n") + 1
        lines[idx] = lines[idx].strip() + f"{miou:.4f},\n"

        # Update mean_f1s
        idx = lines.index("mean_f1s = [\n") + 1
        lines[idx] = lines[idx].strip() + f"{mf1:.4f},\n"

        # Update mean_accs
        idx = lines.index("mean_accs = [\n") + 1
        lines[idx] = lines[idx].strip() + f"{macc:.4f},\n"

        # Update person_ious
        idx = lines.index("person_ious = [\n") + 1
        lines[idx] = lines[idx].strip() + f"{person_iou:.4f},\n"

        # Update person_accs
        idx = lines.index("person_accs = [\n") + 1
        lines[idx] = lines[idx].strip() + f"{person_acc:.4f},\n"

        # Write back to the file
        with open(metrics_file, 'w') as f:
            f.writelines(lines)

    print(f"Metrics summary saved to {metrics_file}")

    print(f"All results saved to {cfg['SAVE_DIR']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)
