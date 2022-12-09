# Dataset-distillation-papers

This repository aims to provide a full list of works about dataset distillation (DD) or dataset condensation (DC).


## Quick links
**Papers sorted by year:** | [2022](#Papers-in-2022-back-to-top) | [2021](#Papers-in-2021-back-to-top) | [2020](#Papers-in-2020-back-to-top) | [2019](#Papers-in-2019-back-to-top) | [2018](#Papers-in-2018-back-to-top) | 



## 2022
### Papers in 2022 [[Back-to-top](#Dataset-distillation-papers)]

| Title     | Type      | Task      | Dataset       | Venue     | Supp. Material     |
|---------|---------|---------|---------|---------|---------|
|(Chen et al) [**Private Set Generation with Discriminative Information**](https://openreview.net/pdf?id=mxnxRw8jiru) |  |  Application: Private Data Generation | MNIST, FashionMNIST | NeurIPS, 2022 | [Code](https://github.com/DingfanChen/Private-Set), [Poster](https://nips.cc/media/PosterPDFs/NeurIPS%202022/53552.png?t=1668599242.828518) |
|(Cui et al) [**DC-BENCH: Dataset Condensation Benchmark**](https://openreview.net/pdf?id=Bs8iFQ7AM6) | Benchmark  | Image Classification |  | NeurIPS, 2022 | [Code](https://dc-bench.github.io/), [Poster](https://nips.cc/media/PosterPDFs/NeurIPS%202022/55673.png?t=1669626268.8753998) |
|[**Dataset Distillation using Neural Feature Regression**](https://openreview.net/pdf?id=2clwrA2tfik) |  | Image Classification | CIFAR100, TinyImageNet, ImageNette and ImageWoof | NeurIPS, 2022 | [Code](https://github.com/yongchao97/FRePo), [Slide](https://docs.google.com/presentation/d/10NMtEVsW-nbEWgbTEJQYMH-rdgOklXZF/edit#slide=id.p3) |
|[**Dataset Distillation via Factorization**](https://openreview.net/pdf?id=luGXvawYWJ) |  | Image Classification | SVHN, CIFAR10/CIFAR100 |  | [Code](https://github.com/Huage001/DatasetFactorization), [Poster](https://nips.cc/media/PosterPDFs/NeurIPS%202022/55231.png?t=1668961755.9041288) |
|[**Efficient Dataset Distillation using Random Feature Approximation**](https://openreview.net/pdf?id=h8Bd7Gm3muB) |  | Image Classification | MNIST, FashionMNIST, SVHN, CIFAR-10/100 | NeurIPS, 2022 | [Code](https://github.com/yolky/RFAD), [Poster](https://nips.cc/media/PosterPDFs/NeurIPS%202022/4be2c8f27b8a420492f2d44463933eb6.png?t=1666483874.2999172) |
|[**Minimizing the Accumulated Trajectory Error to Improve Dataset Distillation**](https://arxiv.org/pdf/2211.11004.pdf) | Accumulated Trajectory Matching |  Image Classification |  | arXiv, Nov., 2022 |  |
|[**Scaling Up Dataset Distillation to ImageNet-1K with Constant Memory**](https://arxiv.org/abs/2211.10586) |  | Image Classification | CIFAR-10/100, ImageNet-1K  | arXiv, Nov., 2022 |  |
|[**Dataset Distillation for Medical Dataset Sharing**](https://arxiv.org/pdf/2209.14603.pdf) | Trajectory Matching |  Application: Medical Data Sharing| COVID-19 Chest X-ray | arXiv, Sept., 2022 |  |
|[**Dataset Distillation using Parameter Pruning**](https://arxiv.org/pdf/2209.14609.pdf) | Parameter Pruning |  Image Classification |  | arXiv, Sept., 2022 |  |
|[**Condensing Graphs via One-Step Gradient Matching**](https://dl.acm.org/doi/abs/10.1145/3534678.3539429?casa_token=hjYiq57R1jcAAAAA:EPtmMLrdCCVYn1Zg1GWq6lVPAIYLOJiv63QE9LODfOGYLvBvRhv7JWsYdxcmW4Hda6t2TwoAewlHrQ) | Gradient Matching | Graph Classification |  | KDD, 2022 | [Code](https://github.com/amazon-science/doscond) |
|[**Dataset Condensation with Latent Space Knowledge Factorization and Sharing**](https://arxiv.org/pdf/2208.10494.pdf) |  | Image Classification |  | arXiv, Aug., 2022 |  |
|[**A Review of Dataset Distillation for Deep Learning**](https://ieeexplore.ieee.org/abstract/document/9932086) | Survey |  Image Classification |  | ICPTS, 2022 |  |
|[**Wearable ImageNet: Synthesizing Tileable Textures via Dataset Distillation**](https://openaccess.thecvf.com/content/CVPR2022W/CVFAD/papers/Cazenavette_Wearable_ImageNet_Synthesizing_Tileable_Textures_via_Dataset_Distillation_CVPRW_2022_paper.pdf) |  | Image Classification |  |CVPRW, 2022 | [Code](https://github.com/GeorgeCazenavette/mtt-distillation) |
|[**Dataset Distillation by Matching Training Trajectories**](https://openaccess.thecvf.com/content/CVPR2022/papers/Cazenavette_Dataset_Distillation_by_Matching_Training_Trajectories_CVPR_2022_paper.pdf) | Trajectory Matching  | Image Classification |  CIFAR-100, Tiny ImageNet, ImageNet subsets | CVPR, 2022 | [Code](https://georgecazenavette.github.io/mtt-distillation/) |
|[**CAFE: Learning to Condense Dataset by Aligning Features**](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_CAFE_Learning_To_Condense_Dataset_by_Aligning_Features_CVPR_2022_paper.pdf) | Feature Alignment |  Image Classification | MNIST, FashionMNIST, SVHN, CIFAR10/100 | CVPR, 2022 | [Code](https://github.com/kaiwang960112/CAFE) |
|[**Dataset Condensation via Efficient Synthetic-Data Parameterization**](https://proceedings.mlr.press/v162/kim22c/kim22c.pdf) |  |  Image Classification |  CIFAR-10, ImageNet, Speech Commands | ICML, 2022 | [Code](https://github.com/snu-mllab/Efficient-Dataset-Condensation) |
|[**Privacy for Free: How does Dataset Condensation Help Privacy?**](https://proceedings.mlr.press/v162/dong22c/dong22c.pdf) | Application: Privacy | Image Classification |  | ICML, 2022 |  |
|[**Graph Condensation for Graph Neural Networks**](https://openreview.net/pdf?id=WLEx3Jo4QaB) | Gradient Matching | Graph Classification | Cora, Citeseer, Ogbn-arxiv; Reddit, Flickr | ICLR, 2022 | [Code](https://github.com/ChandlerBang/GCond) |
|[**Synthesizing Informative Training Samples with GAN**](https://arxiv.org/pdf/2204.07513.pdf) | GAN | Image Classification | CIFAR-10/100  | arXiv, Apr2022  | [Code](https://github.com/VICO-UoE/IT-GAN) |
|[** **]() |  |  Image Classification |  |  | [Code]() |







## 2021
### Papers in 2021 [[Back-to-top](#Dataset-distillation-papers)]
| Title     | Type      | Task      | Dataset       | Venue     | Code      |
|---------|---------|---------|---------|---------|---------|
|[**Dataset Distillation with Infinitely Wide Convolutional Networks**](https://openreview.net/pdf?id=hXWPpJedrVP) | Kernel Ridge Regression |  Image Classification | MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, SVHN | NeurIPS, 2021 | [Code](https://github.com/google-research/google-research/tree/master/kip) |
|[**Dataset Condensation with Distribution Matching**](https://arxiv.org/pdf/2110.04181.pdf) | Distribution Matching | Image Classification | MNIST, CIFAR10/100, TinyImageNet | arXiv, Oct., 2021 | [Code](https://github.com/VICO-UoE/DatasetCondensation) |
|[**Dataset Condensation with Differentiable Siamese Augmentation**](http://proceedings.mlr.press/v139/zhao21a/zhao21a.pdf) | Data Augmentation | Image Classification | MNIST, FashionMNIST, SVHN, CIFAR10/100  | ICML, 2021 | [Code](https://github.com/VICO-UoE/DatasetCondensation), [Video](https://slideslive.com/38958791/dataset-condensation-with-differentiable-siamese-augmentation?ref=recommended) |
|[**Dataset Meta-Learning from Kernel Ridge-Regression**](https://openreview.net/pdf?id=l-PrrQrK0QR) | Kernel Ridge Regression |  Image Classification | MNIST, CIFAR-10 | ICLR, 2021 | [Code](https://github.com/google-research/google-research/tree/master/kip) |
|[**Dataset Condensation with Gradient Matching**](https://openreview.net/pdf?id=mSAKhLYLSsl) | Gradient Matching | Image Classification | CIFAR-10, Fashion-MNIST, MNIST, SVHN, USPS | ICLR, 2021 | [Code](https://github.com/VICO-UoE/DatasetCondensation) |





## 2020
### Papers in 2020 [[Back-to-top](#Dataset-distillation-papers)]
| Title     | Type      | Task      | Dataset       | Venue     | Code      |
|---------|---------|---------|---------|---------|---------|
|[** **]() |  |  Image Classification |  |  | [Code]() |




## 2019
### Papers in 2019 [[Back-to-top](#Dataset-distillation-papers)]
| Title     | Type      | Task      | Dataset       | Venue     | Code      |
|---------|---------|---------|---------|---------|---------|
|[** **]() |  |  Image Classification |  |  | [Code]() |




## 2018
### Papers in 2018 [[Back-to-top](#Dataset-distillation-papers)]

| Title     | Type      | Task      | Dataset       | Venue     | Code      |
|---------|---------|---------|---------|---------|---------|
|[**Dataset Distillation**](https://arxiv.org/pdf/1811.10959.pdf) | Bi-level Opt. | Image Classification | MNIST, CIFAR-10 | arXiv, Nov., 2018 | [Code](https://github.com/SsnL/dataset-distillation) |



|[** **]() |  |  Image Classification |  |  | [Code]() |

