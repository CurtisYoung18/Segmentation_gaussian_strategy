import torch
from torch import nn


class PerturbModelOneTarget(nn.Module):

    def __init__(self, model: nn.Module, pert_ratio: float, target_class_idx: int) -> None:
        super(PerturbModelOneTarget, self).__init__()
        self.model = model
        self.pert_ratio = pert_ratio
        self.target_class_idx = target_class_idx  # Index of 'person' class

    def set_pert_ratio(self, pert_ratio: float):
        self.pert_ratio = pert_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            if isinstance(output, dict):
                logits = output['out']
            else:
                logits = output

        logits_noisy = logits.clone()

        # Extract logits for the 'person' class
        target_logits = logits[:, self.target_class_idx, :, :]

        # Generate positive noise
        noise_std = self.pert_ratio
        target_noise = torch.normal(mean=0.0, std=noise_std, size=target_logits.size()).to(x.device)
        target_noise = target_noise.abs()  # Make noise positive

        # Add noise to the 'person' class logits
        logits_noisy[:, self.target_class_idx, :, :] += target_noise

        return logits_noisy