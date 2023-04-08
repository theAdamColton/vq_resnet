import os
from lightning.pytorch.cli import LightningCLI
import torch

from vq_resnet.model import VQResnet
from vq_resnet.imagenet_datamodule import ImagenetDataModule

torch.set_float32_matmul_precision('medium')


def cli_main():
    cli = LightningCLI(VQResnet, ImagenetDataModule)
    cli.model.save_frozen(os.path.join(cli.trainer.default_root_dir, "frozenmodel.pt"))


if __name__ in {"__main__", "__console__"}:
    cli_main()

