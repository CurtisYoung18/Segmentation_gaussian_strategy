import torch
from torch import nn


class PerturbModelDual(nn.Module):

    def __init__(self, model: nn.Module, pert_ratio: float) -> None:
        super(PerturbModelDual, self).__init__()
        self.model = model
        self.pert_ratio = pert_ratio

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

        noise = torch.normal(mean=0.0, std=self.pert_ratio, size=logits.size()).to(x.device)

        return logits + noise
