import torch
from torchmetrics import Accuracy
import torch.nn as nn
import torchvision.models as models
import lightning as pl

from . import quantizer


class VQResnet(pl.LightningModule):
    def __init__(self,
                 resnet_type:int=34,
                 is_rq=True,
                 quantizer_args={
                     "num_quantizers": 1,
                     "shared_codebook": True,
                     "quantize_dropout" : True,
                     "accept_image_fmap" : True,
                     "codebook_dim": 128,
                     "codebook_size": 256,
                     "decay": 0.8,
                     "eps": 1e-5,
                     "commitment_weight": 0.0,
                     "threshold_ema_dead_code": 0,
                     },
                 # After this block index of resnet quantizer layer will go
                 # All resnet models have 4 main layers. This param puts the
                 # vq layer after the ith layer
                 resnet_insertion_index=3,
                 lr=1e-4,
                 unfreeze_fc=False,
                 unfreeze_resnet_block_indeces=[],

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


        self.in_layers = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            )

        resnet_layers = [
                self.resnet.layer1, 
                self.resnet.layer2, 
                self.resnet.layer3, 
                self.resnet.layer4, 
          ]

        quantizer_args["dim"] = list(list(resnet_layers[resnet_insertion_index-1].children())[-1].children())[-1].num_features

        self.post_q_norm = nn.BatchNorm2d(quantizer_args["dim"])
        #print("quant args", quantizer_args)
        self.quantizer = rq_class(**quantizer_args)

        resnet_layers.insert(resnet_insertion_index, self.quantizer)
        self.resnet_insertion_index = resnet_insertion_index
        self.resnet_layers = nn.Sequential(*resnet_layers)

        # Freezes the preceeding layers to quantizer
        # only the layers following the quantizer will have grad
        for il in self.in_layers:
            il.requires_grad_(False)
        for rl in list(self.resnet_layers.children())[:resnet_insertion_index]:
            rl.requires_grad_(False)
        for rl in list(self.resnet_layers.children())[resnet_insertion_index+1:]:
            rl.requires_grad_(False)
        for unfreeze_idx in unfreeze_resnet_block_indeces:
            list(self.resnet_layers.children())[unfreeze_idx].requires_grad_(True)
        if not unfreeze_fc:
            self.resnet.avgpool.requires_grad_(False)
            self.resnet.fc.requires_grad_(False)

        self.resnet_layers

        self.loss_fn = nn.CrossEntropyLoss()
        self.acc1 = Accuracy("multiclass", num_classes=1000, top_k=1)
        self.acc5 = Accuracy("multiclass", num_classes=1000, top_k=5)

        self.lr = lr

        #print(self.resnet_layers)

        self.save_hyperparameters()


    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

    def encode(self, x, **kwargs):
        """
        kwargs : args for the particular vqlayer
        returns the result of the vqlayer
        """
        x = self.in_layers(x)
        for rl in self.resnet_layers:
            if type(rl) == type(self.quantizer):
                return rl(x, **kwargs)
            else:
                x = rl(x)
        raise Exception("no quantizer in the resnet_layers!")


    def forward(self, x):
        """
        See: https://github.com/pytorch/vision/blob/0387b8821d67ca62d57e3b228ade45371c0af79d/torchvision/models/resnet.py#L166
        """
        x = self.in_layers(x)
        q_loss = 0.0
        for rl in self.resnet_layers:
            if type(rl) == type(self.quantizer):
                x, _, q_loss = rl(x)
                x = self.post_q_norm(x)
                q_loss = q_loss.mean()
            else:
                x = rl(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x, q_loss


    
    def _step(self, batch):
        x, y = batch
        preds, q_loss = self(x)

        classification_loss = self.loss_fn(preds, y)
        acc1, acc5 = self.acc1(preds, y), self.acc5(preds, y)
        return classification_loss, q_loss, acc1, acc5

    def training_step(self, batch, batch_idx):
        c_loss, q_loss, acc1, acc5 = self._step(batch)
        # perform logging
        self.log_dict(
                {"t_c_loss":c_loss,
                 "t_q_loss":q_loss,
                 "t_acc_1": acc1,
                 "t_acc_5": acc5,
                 }, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return c_loss + q_loss


    def validation_step(self, batch, _):
        c_loss, q_loss, acc_1, acc_5 = self._step(batch)
        # perform logging
        self.log_dict(
                {"v_c_loss":c_loss,
                 "v_q_loss":q_loss,
                 "v_acc_1": acc_1,
                 "v_acc_5": acc_5,
                 }, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return c_loss + q_loss

