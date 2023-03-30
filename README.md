# Vector Quantize Resnet

Contains training code to fine tune pretrained Resnet models with a vq layer inserted somewhere into the Resnet model.

The purpose of this is to use the tokens from the VQ layer in downstream applications.

Find some pretrained model [here](https://huggingface.co/adams-story/vq_resnet)

Thanks to lucidrains for [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)

# Notes:

If you train with low or zero commitment loss, and you unfreeze resnet layers after the quantization, the residual layer may start to behave strangely. You'd expect that after the initial estimation of z ~ k\_0, adding more residual quantization steps will make z hat closer (in euclid) to z. This is not the case; the model will not be incentivized to make the quantization step produce outputs close to the inputs.
