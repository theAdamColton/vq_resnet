from lightning.pytorch.cli import LightningCLI

from vq_resnet.model import VQResnet
from vq_resnet.imagenet_datamodule import ImagenetDataModule


def cli_main():
    cli = LightningCLI(VQResnet, ImagenetDataModule)


if __name__ in {"__main__", "__console__"}:
    cli_main()

