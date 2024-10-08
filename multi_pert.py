# 双边加噪 人大车树小

import torch
from torch import nn

class PerturbModelTwo(nn.Module):
    def __init__(self, model: nn.Module, pert_ratio: float, target_perturbations: list) -> None:
        super(PerturbModelTwo, self).__init__()
        self.model = model
        self.pert_ratio = pert_ratio
        self.target_perturbations = target_perturbations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            if isinstance(output, dict):
                logits = output['out']
            else:
                logits = output

        logits_noisy = logits.clone()

        for perturb in self.target_perturbations:
            target_idx = perturb['index']
            positive_noise = perturb['positive']

            # 提取目标类别的 logits
            target_logits = logits[:, target_idx, :, :]

            # 确定噪声的标准差
            noise_std = self.pert_ratio

            # 生成零均值、高斯分布的噪声
            target_noise = torch.normal(mean=0.0, std=noise_std, size=target_logits.size()).to(x.device)

            # 根据 positive_noise 参数调整噪声方向
            if positive_noise:
                target_noise = target_noise.abs()
            else:
                target_noise = -target_noise.abs()

            # 将噪声添加到目标类别的 logits 上
            logits_noisy[:, target_idx, :, :] += target_noise

        return logits_noisy