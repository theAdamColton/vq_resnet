# Vector Quantize Resnet

Contains training code to fine tune pretrained Resnet models with a vq layer inserted somewhere into the Resnet model.

The purpose of this is to use the tokens from the VQ layer in downstream applications.

Find some pretrained model [here](https://huggingface.co/adams-story/vq_resnet)

Thanks to lucidrains for [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
