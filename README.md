# Compressed Multi-Modality Face Anti-Spoofing Model
A final year project proposing an approach to compress existing state of the art face anti spoofing models for deployment to edge devices

To date , there are no other method or research in currently available literature that tries to perform model compression on face anti-spoofing models . This approach combines two existing methods available in current research to produce a lightweight-model that can perform face anti-spoofing.

1. The network design in https://arxiv.org/abs/2103.00948 is implemented as a basic model first since it presents state-of-the-art performance as of the start of this project. It uses a multi-head architecture composed of 8 DenseNet blocks and supervised by a novel Cross Modal Focal Loss function as proposed by the authors. It can learn from 2 modalities with an option to extend to more modalities.
2. The network is then replaced with GhostNet (https://arxiv.org/abs/1911.11907) as a form of model compression , which makes the model consume 97% less computational resources (0.31G FLOPS) and performs 67% faster (2.3ms inference time)
