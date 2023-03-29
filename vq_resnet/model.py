import torch
from torchmetrics import Accuracy
import torch.nn as nn
import torchvision
import torchinfo
import torchvision.models as models
import lightning as pl

from . import quantizer


class VQResnet(pl.LightningModule):
    def __init__(self,
                 resnet_type:int=34,
                 is_rq=True,
                 quantizer_args={},
                 # After this block index of resnet quantizer layer will go
                 # All resnet models have 4 main layers. This param puts the
                 # vq layer after the ith layer
                 resnet_insertion_index=3,
                 ):
        super().__init__()

        resnets = {
            18: (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
            34: (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
            50: (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
            101: (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2),
            152: (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2),
        }
        _rm, _rw = resnets[resnet_type]
        self.resnet = _rm(_rw)

        self.is_rq = is_rq
        rq_class = quantizer.VectorQuantize if not is_rq else quantizer.ResidualVQ

        self.quantizer = rq_class(**quantizer_args)

        self.in_layers = nn.ModuleList([
            self.resenet.conv1,
            self.resenet.bn1,
            self.resenet.relu,
            self.resenet.maxpool,
        ])

        self.resnet_layers = nn.ModuleList([
                self.resnet.layer1, 
                self.resnet.layer2, 
                self.resnet.layer3, 
                self.resnet.layer4, 
              ])

        self.resnet_layers.insert(resnet_insertion_index, self.quantizer)

        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy("multiclass", 1000)

        torchinfo.summary(self, (7, 3, 224, 224))


    def forward(self, x):
        """
        See: https://github.com/pytorch/vision/blob/0387b8821d67ca62d57e3b228ade45371c0af79d/torchvision/models/resnet.py#L166
        """
        x = self.in_layers(x)
        x = self.resnet_layers(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x


    
    def _step(self, batch):
        x, y = batch
        preds = self(x)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()

        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss


    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

