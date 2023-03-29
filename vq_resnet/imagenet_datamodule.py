"""
patched together from https://github.com/Lightning-AI/lightning-bolts/
"""
import os
import gzip
import os
import tarfile 
import zipfile
import shutil
import tempfile
from contextlib import contextmanager
import hashlib
import numpy as np
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageNet
from torchvision.datasets.imagenet import load_meta_file
from torch.utils.data import DataLoader
from typing import Any, Callable, Optional, List
from lightning.pytorch import LightningDataModule


def imagenet_normalization() -> Callable:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return normalize

def is_within_directory(directory: str, target: str) -> bool:
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == abs_directory

def safe_extract_tarfile(
    tar: tarfile.TarFile,
    path: str = ".",
    members: Optional[List[tarfile.TarInfo]] = None,
    *,
    numeric_owner: bool = False,
) -> None:
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise RuntimeError(f"Attempted Path Traversal in Tar File {tar.name} with member: {member.name}")

    tar.extractall(path, members, numeric_owner=numeric_owner)

def _extract_tar(from_path: str, to_path: str) -> None:
    with tarfile.open(from_path, "r:*") as tar:
        safe_extract_tarfile(tar, path=to_path)


def _extract_gzip(from_path: str, to_path: str) -> None:
    to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
    with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
        out_f.write(zip_f.read())


def _extract_zip(from_path: str, to_path: str) -> None:
    with zipfile.ZipFile(from_path, "r") as z:
        z.extractall(to_path)


def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> None:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    extracted = False
    for fn in (_extract_tar, _extract_gzip, _extract_zip):
        try:
            fn(from_path, to_path)
            extracted = True
            break
        except (tarfile.TarError, zipfile.BadZipfile, OSError):
            continue

    if not extracted:
        raise ValueError(f"Extraction of {from_path} not supported")

    if remove_finished:
        os.remove(from_path)


class UnlabeledImagenet(ImageNet):
    """Official train set gets split into train, val. (using nb_imgs_per_val_class for each class). Official
    validation becomes test set.
    Within each class, we further allow limiting the number of samples per class (for semi-sup lng)
    """

    def __init__(
        self,
        root,
        split: str = "train",
        num_classes: int = -1,
        num_imgs_per_class: int = -1,
        num_imgs_per_class_val_split: int = 50,
        meta_dir=None,
        **kwargs,
    ):
        """
        Args:
            root: path of dataset
            split:
            num_classes: Sets the limit of classes
            num_imgs_per_class: Limits the number of images per class
            num_imgs_per_class_val_split: How many images per class to generate the val split
            download:
            kwargs:
        """

        root = self.root = os.path.expanduser(root)

        # [train], [val] --> [train, val], [test]
        original_split = split
        if split == "train" or split == "val":
            split = "train"

        if split == "test":
            split = "val"

        self.split = split
        split_root = os.path.join(root, split)
        meta_dir = meta_dir if meta_dir is not None else split_root
        wnid_to_classes = load_meta_file(meta_dir)[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        # shuffle images first
        np.random.seed(1234)
        np.random.shuffle(self.imgs)

        # partition train set into [train, val]
        if split == "train":
            train, val = self.partition_train_set(self.imgs, num_imgs_per_class_val_split)
            if original_split == "train":
                self.imgs = train
            if original_split == "val":
                self.imgs = val

        # limit the number of images in train or test set since the limit was already applied to the val set
        if split in ["train", "test"]:
            if num_imgs_per_class != -1:
                clean_imgs = []
                cts = {x: 0 for x in range(len(self.classes))}
                for img_name, idx in self.imgs:
                    if cts[idx] < num_imgs_per_class:
                        clean_imgs.append((img_name, idx))
                        cts[idx] += 1

                self.imgs = clean_imgs

        # limit the number of classes
        if num_classes != -1:
            # choose the classes at random (but deterministic)
            ok_classes = list(range(num_classes))
            np.random.seed(1234)
            np.random.shuffle(ok_classes)
            ok_classes = ok_classes[:num_classes]
            ok_classes = set(ok_classes)

            clean_imgs = []
            for img_name, idx in self.imgs:
                if idx in ok_classes:
                    clean_imgs.append((img_name, idx))

            self.imgs = clean_imgs

        # shuffle again for final exit
        np.random.seed(1234)
        np.random.shuffle(self.imgs)

        # list of class_nbs for each image
        idcs = [idx for _, idx in self.imgs]

        self.wnids = self.classes
        self.wnid_to_idx = {wnid: idx for idx, wnid in zip(idcs, self.wnids)}
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for clss, idx in zip(self.classes, idcs) for cls in clss}

        # update the root data
        self.samples = self.imgs
        self.targets = [s[1] for s in self.imgs]

    def partition_train_set(self, imgs, nb_imgs_in_val):
        val = []
        train = []

        cts = {x: 0 for x in range(len(self.classes))}
        for img_name, idx in imgs:
            if cts[idx] < nb_imgs_in_val:
                val.append((img_name, idx))
                cts[idx] += 1
            else:
                train.append((img_name, idx))

        return train, val

    @classmethod
    def generate_meta_bins(cls, devkit_dir):
        files = os.listdir(devkit_dir)
        if "ILSVRC2012_devkit_t12.tar.gz" not in files:
            raise FileNotFoundError(
                "devkit_path must point to the devkit file"
                "ILSVRC2012_devkit_t12.tar.gz. Download from here:"
                "http://www.image-net.org/challenges/LSVRC/2012/downloads"
            )

        parse_devkit_archive(devkit_dir)
        print(f"meta.bin generated at {devkit_dir}/meta.bin")



def _verify_archive(root, file, md5):
    if not _check_integrity(os.path.join(root, file), md5):
        raise RuntimeError(
            f"The archive {file} is not present in the root directory or is corrupted."
            f" You need to download it externally and place it in {root}."
        )



def _check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return _check_md5(fpath, md5)



def _check_md5(fpath, md5, **kwargs):
    return md5 == _calculate_md5(fpath, **kwargs)



def _calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()



def parse_devkit_archive(root, file=None):
    """Parse the devkit archive of the ImageNet2012 classification dataset and save the meta information in a
    binary file.
    Args:
        root (str): Root directory containing the devkit archive
        file (str, optional): Name of devkit archive. Defaults to
            'ILSVRC2012_devkit_t12.tar.gz'
    """
    from scipy import io as sio

    def parse_meta_mat(devkit_root):
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(", ")) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root):
        file = os.path.join(devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt")
        with open(file) as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @contextmanager
    def get_tmp_dir():
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)

    archive_meta = ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf")
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    with get_tmp_dir() as tmp_dir:
        extract_archive(os.path.join(root, file), tmp_dir)

        devkit_root = os.path.join(tmp_dir, "ILSVRC2012_devkit_t12")
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        META_FILE = "meta.bin"

        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))



class ImagenetDataModule(LightningDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/
        Sample-of-Images-from-the-ImageNet-Dataset-used-in-the-ILSVRC-Challenge.png
        :width: 400
        :alt: Imagenet
    Specs:
        - 1000 classes
        - Each image is (3 x varies x varies) (here we default to 3 x 224 x 224)
    Imagenet train, val and test dataloaders.
    The train set is the imagenet train.
    The val set is taken from the train set with `num_imgs_per_val_class` images per class.
    For example if `num_imgs_per_val_class=2` then there will be 2,000 images in the validation set.
    The test set is the official imagenet validation set.
     Example::
        from pl_bolts.datamodules import ImagenetDataModule
        dm = ImagenetDataModule(IMAGENET_PATH)
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    """

    name = "imagenet"

    def __init__(
        self,
        data_dir: str,
        meta_dir: Optional[str] = None,
        num_imgs_per_val_class: int = 50,
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
            meta_dir: path to meta.bin file
            num_imgs_per_val_class: how many images per class for the validation set
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
        self.meta_dir = meta_dir
        self.num_imgs_per_val_class = num_imgs_per_val_class
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_samples = 1281167 - self.num_imgs_per_val_class * self.num_classes

    @property
    def num_classes(self) -> int:
        """
        Return:
            1000
        """
        return 1000

    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:
        """This method already assumes you have imagenet2012 downloaded. It validates the data using the meta.bin.
        .. warning:: Please download imagenet on your own first.
        """
        self._verify_splits(self.data_dir, "train")
        self._verify_splits(self.data_dir, "val")

        for split in ["train", "val"]:
            files = os.listdir(os.path.join(self.data_dir, split))
            if "meta.bin" not in files:
                raise FileNotFoundError(
                    """
    no meta.bin present. Imagenet is no longer automatically downloaded by PyTorch.
    To get imagenet:
    1. download the devkit (https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz)
    2. generate the meta.bin file using the devkit
    3. copy the meta.bin file into both train and val split folders
    To generate the meta.bin do the following:
    from pl_bolts.datasets import UnlabeledImagenet
    path = '/path/to/folder/with/ILSVRC2012_devkit_t12.tar.gz/'
    UnlabeledImagenet.generate_meta_bins(path)
                """
                )

    def train_dataloader(self) -> DataLoader:
        """Uses the train split of imagenet2012 and puts away a portion of it for the validation split."""
        transforms = self.train_transform()

        dataset = UnlabeledImagenet(
            self.data_dir,
            num_imgs_per_class=-1,
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            meta_dir=self.meta_dir,
            split="train",
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
        """Uses the part of the train split of imagenet2012  that was not used for training via
        `num_imgs_per_val_class`
        Args:
            batch_size: the batch size
            transforms: the transforms
        """
        transforms = self.val_transform()

        dataset = UnlabeledImagenet(
            self.data_dir,
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            meta_dir=self.meta_dir,
            split="val",
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

        dataset = UnlabeledImagenet(
            self.data_dir, num_imgs_per_class=-1, meta_dir=self.meta_dir, split="test", transform=transforms
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
                transforms.RandomHorizontalFlip(),
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
