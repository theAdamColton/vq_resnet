"""
patched together from https://github.com/Lightning-AI/lightning-bolts/
"""
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Any, Callable
from lightning.pytorch import LightningDataModule


def imagenet_normalization() -> Callable:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return normalize


class ImagenetDataModule(LightningDataModule):
    name = "imagenet"

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        num_workers: int = 0,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset file
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)


        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        """
        Return:
            1000
        """
        return 1000


    def train_dataloader(self) -> DataLoader:
        """Uses the train split of imagenet2012 and puts away a portion of it for the validation split."""
        transforms = self.train_transform()

        dataset = ImageFolder(
            self.data_dir,
            transform=transforms,
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        transforms = self.val_transform()

        dataset = ImageFolder(
            self.data_dir,
            transform=transforms,
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        """Uses the validation split of imagenet2012 for testing."""
        transforms = self.val_transform() 

        dataset = ImageFolder(
            self.data_dir, transform=transforms
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def train_transform(self) -> Callable:
        """The standard imagenet transforms.
        .. code-block:: python
            transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """
        preprocessing = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                #transforms.RandAugment(9),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

        return preprocessing

    def val_transform(self) -> Callable:
        """The standard imagenet transforms for validation.
        .. code-block:: python
            transforms.Compose([
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """

        preprocessing = transforms.Compose(
            [
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )
        return preprocessing
