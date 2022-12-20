# Dataset-distillation-papers

This repository aims to provide a full list of works about dataset distillation (DD) or dataset condensation (DC).


## Quick links
**Papers sorted by year:** | [2022](#Papers-in-2022-back-to-top) | [2021](#Papers-in-2021-back-to-top) | [2020](#Papers-in-2020-back-to-top) | [2019](#Papers-in-2019-back-to-top) | [2018](#Papers-in-2018-back-to-top) | 



## 2022
### Papers in 2022 [[Back-to-top](#Dataset-distillation-papers)]

| Author    | Title     | Type      | Task      | Dataset       | Venue     | Supp. Material     |
|---------|---------|---------|---------|---------|---------|---------|
| Guang Li et al|[**Compressed Gastric Image Generation Based on Soft-Label Dataset Distillation for Medical Data Sharing**](https://www.sciencedirect.com/science/article/pii/S0169260722005703) | Soft-Label Distillation |  Application: Medical Data Sharing | Gastric X-ray | Computer Methods and Programs in Biomedicine, 2022 |  |
| Zijia Wang et al |[**Gift from nature: Potential Energy Minimization for explainable dataset distillation**](https://openaccess.thecvf.com/content/ACCV2022W/MLCSA/papers/Wang_Gift_from_nature_Potential_Energy_Minimization_for_explainable_dataset_distillation_ACCVW_2022_paper.pdf) | Potential Energy Minimization |  Image Classification | miniImageNet, CUB-200 | ACCV Workshop, 2022 |  |
| Zhiwei Deng et al |[**Remember the Past: Distilling Datasets into Addressable Memories for Neural Networks**](https://openreview.net/pdf?id=RYZyj_wwgfa) |  |  Image Classification |  MNIST, SVHN, CIFAR10/100, TinyImageNet | NeurIPS, 2022 | [Code](https://github.com/princetonvisualai/RememberThePast-DatasetDistillation) |
| Noveen Sachdeva et al |[**Infinite Recommendation Networks: A Data-Centric Approach**](https://arxiv.org/pdf/2206.02626.pdf) | Neural Tangent Kernel | Application: Recommender System  | Amazon Magazine, ML-1M, Douban, Netflix | NeurIPS, 2022 | [Code](https://github.com/noveens/distill_cf) |
| Dingfang Chen et al | [**Private Set Generation with Discriminative Information**](https://openreview.net/pdf?id=mxnxRw8jiru) |  |  Application: Private Data Generation | MNIST, FashionMNIST | NeurIPS, 2022 | [Code](https://github.com/DingfanChen/Private-Set), [Poster](https://nips.cc/media/PosterPDFs/NeurIPS%202022/53552.png?t=1668599242.828518) |
| Justin Cui et al | [**DC-BENCH: Dataset Condensation Benchmark**](https://openreview.net/pdf?id=Bs8iFQ7AM6) | Benchmark  | Image Classification |  | NeurIPS, 2022 | [Code](https://dc-bench.github.io/), [Poster](https://nips.cc/media/PosterPDFs/NeurIPS%202022/55673.png?t=1669626268.8753998) |
|Yongchao Zhou et al |[**Dataset Distillation using Neural Feature Regression**](https://openreview.net/pdf?id=2clwrA2tfik) |  | Image Classification | CIFAR100, TinyImageNet, ImageNette, ImageWoof | NeurIPS, 2022 | [Code](https://github.com/yongchao97/FRePo), [Slide](https://docs.google.com/presentation/d/10NMtEVsW-nbEWgbTEJQYMH-rdgOklXZF/edit#slide=id.p3) |
| Songhua Liu et al |[**Dataset Distillation via Factorization**](https://openreview.net/pdf?id=luGXvawYWJ) |  | Image Classification | SVHN, CIFAR10/100 |  | [Code](https://github.com/Huage001/DatasetFactorization), [Poster](https://nips.cc/media/PosterPDFs/NeurIPS%202022/55231.png?t=1668961755.9041288) |
| Noel Loo et al|[**Efficient Dataset Distillation using Random Feature Approximation**](https://openreview.net/pdf?id=h8Bd7Gm3muB) |  | Image Classification | MNIST, FashionMNIST, SVHN, CIFAR-10/100 | NeurIPS, 2022 | [Code](https://github.com/yolky/RFAD), [Poster](https://nips.cc/media/PosterPDFs/NeurIPS%202022/4be2c8f27b8a420492f2d44463933eb6.png?t=1666483874.2999172) |
| Yihan Wu et al |[**Towards Robust Dataset Learning**](https://arxiv.org/pdf/2211.10752.pdf) | Tri-level Optimization |  Robust Image Classification | MNIST, CIFAR10, TinyImageNet | arXiv, Nov., 2022 |  |
| Andrey Zhmoginov et al |[**Decentralized Learning with Multi-Headed Distillation**](https://arxiv.org/pdf/2211.15774.pdf) | Local DD |  Application: FL | CIFAR-10/100 | arXiv, Nov., 2022 |  |
| Jiawei Du et al |[**Minimizing the Accumulated Trajectory Error to Improve Dataset Distillation**](https://arxiv.org/pdf/2211.11004.pdf) | Accumulated Trajectory Matching |  Image Classification |  | arXiv, Nov., 2022 |  |
| Justin Cui et al |[**Scaling Up Dataset Distillation to ImageNet-1K with Constant Memory**](https://arxiv.org/abs/2211.10586) |  | Image Classification | CIFAR-10/100, ImageNet-1K  | arXiv, Nov., 2022 |  |
| Renjie Pi et al |[**DYNAFED: Tackling Client Data Heterogeneity with Global Dynamics**](https://arxiv.org/pdf/2211.10878.pdf) |  |  Application: FL | FMNIST, CIFAR10, CINIC10 | arXiv, Nov., 2022 |  |
| Zongwei Wang et al |[**Quick Graph Conversion for Robust Recommendation**](https://arxiv.org/pdf/2210.10321.pdf) | Gradient Matching |  Application: Recommender System | Beauty, Alibaba-iFashion, Yelp2018 | arXiv, Oct., 2022 | |
| Yanlin Zhou et al |[**Communication-Efficient and Attack-Resistant Federated Edge Learning with Dataset Distillation**](https://ieeexplore.ieee.org/abstract/document/9925087) | MNIST, Landmark, IMDB, etc |  Application: FL |  | IEEE TCC, 2022 | [Code]() |
<!-- | Justin Cui et al |[**Memory-efficient Trajectory Matching for Scalable Dataset Distillation**](https://openreview.net/pdf?id=dN70O8pmW8) | Memory-efficient Trajectory Matching |  Image Classification | CIFAR-10/100, ImageNet-1K | OpenReview, Sept., 2022 | [Code](https://openreview.net/forum?id=dN70O8pmW8) | -->
| Nicholas Carlini et al |[**No Free Lunch in "Privacy for Free: How does Dataset Condensation Help Privacy"**](https://arxiv.org/abs/2209.14987) |  |  Application: Privacy | CIFAR-10 | arXiv, Sept., 2022 |  |
| Guang Li et al |[**Dataset Distillation for Medical Dataset Sharing**](https://arxiv.org/pdf/2209.14603.pdf) | Trajectory Matching |  Application: Medical Data Sharing| COVID-19 Chest X-ray | arXiv, Sept., 2022 |  |
| Guang Li et al |[**Dataset Distillation using Parameter Pruning**](https://arxiv.org/pdf/2209.14609.pdf) | Parameter Pruning |  Image Classification |  CIFAR-10/100 | arXiv, Sept., 2022 |  |
| Ping Liu et al |[**Meta Knowledge Condensation for Federated Learning**](https://arxiv.org/abs/2209.14851) |  |  Application: FL | MNIST | arXiv, Sept., 2022 |  |
| Dmitry Medvedev et al |[**Learning to Generate Synthetic Training Data Using Gradient Matching and Implicit Differentiation**](https://link.springer.com/chapter/10.1007/978-3-031-15168-2_12) | Gradient Matching, Implicit Differentiation |  Image Classification | MNIST | CCIS, 2022 | [Code](https://github.com/dm-medvedev/EfficientDistillation) |
| Wei Jin et al |[**Condensing Graphs via One-Step Gradient Matching**](https://dl.acm.org/doi/abs/10.1145/3534678.3539429?casa_token=hjYiq57R1jcAAAAA:EPtmMLrdCCVYn1Zg1GWq6lVPAIYLOJiv63QE9LODfOGYLvBvRhv7JWsYdxcmW4Hda6t2TwoAewlHrQ) | Gradient Matching | Graph Classification |  | KDD, 2022 | [Code](https://github.com/amazon-science/doscond) |
| Rui Song et al |[**Federated Learning via Decentralized Dataset Distillation in Resource-Constrained Edge Environments**](https://arxiv.org/abs/2208.11311) | Local DD |  Application: FL | MNIST, CIFAR10 | arXiv, Aug., 2022 |  |
| Hae Beom Lee et al |[**Dataset Condensation with Latent Space Knowledge Factorization and Sharing**](https://arxiv.org/pdf/2208.10494.pdf) | Local DD | Image Classification |  | arXiv, Aug., 2022 |  |
| Thi-Thu-Huong Le et al |[**A Review of Dataset Distillation for Deep Learning**](https://ieeexplore.ieee.org/abstract/document/9932086) | Survey |  Image Classification |  | ICPTS, 2022 |  |
| Zixuan Jiang et al |[**Delving into Effective Gradient Matching for Dataset Condensation**](https://arxiv.org/pdf/2208.00311.pdf) | Gradient Matching |  Image Classification |  MNIST/FashionMNIST, SVHN, CIFAR-10/100. | arXiv, Jul., 2022 | [Code]() |
| Yuanhao Xiong et al |[**FedDM: Iterative Distribution Matching for Communication-Efficient Federated Learning**](https://arxiv.org/pdf/2207.09653.pdf) |  |  Application: FL | MNIST, CIFAR10/100 | arXiv, Jul., 2022 |  |
| Nadiya Shvai et al |[**DEvS: Data Distillation Algorithm Based on Evolution Strategy**](https://dl.acm.org/doi/pdf/10.1145/3520304.3528819) | Evolution Strategy |  Image Classification |  CIFAR-10 | GECCO, 2022 |  |
| Mattia Sangermano |[**Sample Condensation in Online Continual Learning**](https://ieeexplore.ieee.org/abstract/document/9892299/) | Gradient Matching |  Application: Continual learning | SplitMNIST, SplitFashionMNIST, SplitCIFAR10 | IJCNN, 2022 | [Code](https://github.com/MattiaSangermano/OLCGM) |
| George Cazenavette et al |[**Wearable ImageNet: Synthesizing Tileable Textures via Dataset Distillation**](https://openaccess.thecvf.com/content/CVPR2022W/CVFAD/papers/Cazenavette_Wearable_ImageNet_Synthesizing_Tileable_Textures_via_Dataset_Distillation_CVPRW_2022_paper.pdf) |  | Image Classification |  |CVPRW, 2022 | [Code](https://github.com/GeorgeCazenavette/mtt-distillation) |
| George Cazenavette et al |[**Dataset Distillation by Matching Training Trajectories**](https://openaccess.thecvf.com/content/CVPR2022/papers/Cazenavette_Dataset_Distillation_by_Matching_Training_Trajectories_CVPR_2022_paper.pdf) | Trajectory Matching  | Image Classification |  CIFAR-100, Tiny ImageNet, ImageNet subsets | CVPR, 2022 | [Code](https://georgecazenavette.github.io/mtt-distillation/) |
| Kai Wang et al |[**CAFE: Learning to Condense Dataset by Aligning Features**](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_CAFE_Learning_To_Condense_Dataset_by_Aligning_Features_CVPR_2022_paper.pdf) | Feature Alignment |  Image Classification | MNIST, FashionMNIST, SVHN, CIFAR10/100 | CVPR, 2022 | [Code](https://github.com/kaiwang960112/CAFE) |
| Mengyang Liu et al |[**Graph Condensation via Receptive Field Distribution Matching**](https://arxiv.org/pdf/2206.13697.pdf) | Rceptive Field Distribution Matching |  Graph Classification | Cora, PubMed, Citeseer, Ogbn-arxiv, Flikcr | arXiv, Jun., 2022 |  |
| Saehyung Lee et al |[**Dataset Condensation with Contrastive Signals**](https://proceedings.mlr.press/v162/lee22b/lee22b.pdf) | Contrastive Learning |  Image Classification | SVHN, CIFAR-10/100; Automobile, Terrier, Fish | ICML 2022 | [Code](https://github.com/Saehyung-Lee/DCC) |
| Jang-Hyun Kim et al |[**Dataset Condensation via Efficient Synthetic-Data Parameterization**](https://proceedings.mlr.press/v162/kim22c/kim22c.pdf) |  |  Image Classification |  CIFAR-10, ImageNet, Speech Commands | ICML, 2022 | [Code](https://github.com/snu-mllab/Efficient-Dataset-Condensation) |
| Tian Dong et al |[**Privacy for Free: How does Dataset Condensation Help Privacy?**](https://proceedings.mlr.press/v162/dong22c/dong22c.pdf) | Application: Privacy | Image Classification |  | ICML, 2022 |  |
| Wei Jin et al |[**Graph Condensation for Graph Neural Networks**](https://openreview.net/pdf?id=WLEx3Jo4QaB) | Gradient Matching | Graph Classification | Cora, Citeseer, Ogbn-arxiv; Reddit, Flickr | ICLR, 2022 | [Code](https://github.com/ChandlerBang/GCond) |
| Bo Zhao et al |[**Synthesizing Informative Training Samples with GAN**](https://arxiv.org/pdf/2204.07513.pdf) | GAN | Image Classification | CIFAR-10/100  | arXiv, Apr. 2022  | [Code](https://github.com/VICO-UoE/IT-GAN) |
| Shengyuan Hu et al |[**FedSynth: Gradient Compression via Synthetic Data in Federated Learning**](https://arxiv.org/pdf/2204.01273.pdf) |  | Application: FL | MNIST, FEMNIST, Reddit |  |  |
| Aminu Musa et al |[**Learning from Small Datasets: An Efficient Deep Learning Model for Covid-19 Detection from Chest X-ray Using Dataset Distillation Technique**](https://ieeexplore.ieee.org/abstract/document/9803131) |  | Application: Medical Imaging | Chest X-ray | NIGERCON, 2022 |  |
| Seong-Woong Kim et al |[**Stable Federated Learning with Dataset Condensation**](http://jcse.kiise.org/files/V16N1-05.pdf) |  |  Application: FL | CIFAR-10 | JCSE, 2022 |  |
| Robin T. Schirrmeister et al |[**When less is more: Simplifying inputs aids neural network understanding**](https://arxiv.org/pdf/2201.05610.pdf) |  |  Application: Understanding NN | MNIST, Fashion-MNIST, CIFAR10/100, | arXiv, Jan, 2022 | |
| |[** **]() |  |  Image Classification |  |  | [Code]() |







## 2021
### Papers in 2021 [[Back-to-top](#Dataset-distillation-papers)]
| Author    | Title     | Type      | Task      | Dataset       | Venue     | Supp. Material     |
|---------|---------|---------|---------|---------|---------|---------|
| Timothy Nguyen et al |[**Dataset Distillation with Infinitely Wide Convolutional Networks**](https://openreview.net/pdf?id=hXWPpJedrVP) | Kernel Ridge Regression |  Image Classification | MNIST, Fashion-MNIST, CIFAR-10/100, SVHN | NeurIPS, 2021 | [Code](https://github.com/google-research/google-research/tree/master/kip) |
| Bo Zhao et al |[**Dataset Condensation with Distribution Matching**](https://arxiv.org/pdf/2110.04181.pdf) | Distribution Matching | Image Classification | MNIST, CIFAR10/100, TinyImageNet | arXiv, Oct., 2021 | [Code](https://github.com/VICO-UoE/DatasetCondensation) |
| Ilia Sucholutsky et al |[**Soft-Label Dataset Distillation and Text Dataset Distillation**](https://ieeexplore.ieee.org/abstract/document/9533769) | Label Distillation |  Image/Text Classification | MNIST, IMDB | IJCNN, 2021 | [Code](https://github.com/ilia10000/dataset-distillation) |
| Felix Wiewel et al |[**Condensed Composite Memory Continual Learning**](https://ieeexplore.ieee.org/abstract/document/9533491/) | Gradient Matching |  Application: Continual Learning |  | IJCNN, 2021 | [Code](https://github.com/FelixWiewel/CCMCL) |
| Bo Zhao et al |[**Dataset Condensation with Differentiable Siamese Augmentation**](http://proceedings.mlr.press/v139/zhao21a/zhao21a.pdf) | Data Augmentation | Image Classification | MNIST, FashionMNIST, SVHN, CIFAR10/100  | ICML, 2021 | [Code](https://github.com/VICO-UoE/DatasetCondensation), [Video](https://slideslive.com/38958791/dataset-condensation-with-differentiable-siamese-augmentation?ref=recommended) |
| Timothy Nguyen et al |[**Dataset Meta-Learning from Kernel Ridge-Regression**](https://openreview.net/pdf?id=l-PrrQrK0QR) | Kernel Ridge Regression |  Image Classification | MNIST, CIFAR-10 | ICLR, 2021 | [Code](https://github.com/google-research/google-research/tree/master/kip) |
| Bo Zhao et al |[**Dataset Condensation with Gradient Matching**](https://openreview.net/pdf?id=mSAKhLYLSsl) | Gradient Matching | Image Classification | CIFAR-10, Fashion-MNIST, MNIST, SVHN, USPS | ICLR, 2021 | [Code](https://github.com/VICO-UoE/DatasetCondensation) |
| |[**New Properties of the Data Distillation Method When Working with Tabular Data**](https://link.springer.com/chapter/10.1007/978-3-030-72610-2_29) | Simulation |  Tabular Classification |  | LNISA, 2021 | [Code](https://github.com/dm-medvedev/dataset-distillation) |
| Yongqi Li et al |[**Data Distillation for Text Classification**](https://arxiv.org/abs/2104.08448) |  |  Text Classification |  | arXiv, Apr., 2021 | [Code]() |
| Ilia Sucholutsky et al |[**‘Less Than One’-Shot Learning: Learning N Classes From M<N Samples**](https://ojs.aaai.org/index.php/AAAI/article/view/17171) | Label Distillation |  Application: Few-Shot Learning | Similation | AAAI, 2021 | [Code](https://github.com/ilia10000/LO-Shot) |




## 2020
### Papers in 2020 [[Back-to-top](#Dataset-distillation-papers)]
| Author    | Title     | Type      | Task      | Dataset       | Venue     | Supp. Material     |
|---------|---------|---------|---------|---------|---------|---------|
| Ondrej Bohdal et al |[**Flexible Dataset Distillation: Learn Labels Instead of Images**](https://arxiv.org/pdf/2006.08572.pdf) | Label Distillation |  Image Classification |  MNIST, CIFAR-10/100, CUB | NeurIPS workshop, 2020 | [Code](https://github.com/ondrejbohdal/label-distillation) |
| Chengeng Huang et al |[**Generative Dataset Distillation**](https://ieeexplore.ieee.org/abstract/document/9546880) | Generative Adversarial Networks | MNIST | Image Classification  | BigCom, 2021 | |
| Jack Goetz et al |[**Federated Learning via Synthetic Data**](https://arxiv.org/pdf/2008.04489.pdf) | Bi-level Optimization |  Application: FL |  | arXiv, Aug., 2020  |  |
| Guang Li et al |[**Soft-Label Anonymous Gastric X-Ray Image Distillation**](https://ieeexplore.ieee.org/abstract/document/9191357) | Label Distillation |  Application: Medical Data Sharing | X-ray Images | ICIP, 2020 |  |

| |[** **]() |  |  Image Classification |  |  | [Code]() |



## 2019
### Papers in 2019 [[Back-to-top](#Dataset-distillation-papers)]
| Author    | Title     | Type      | Task      | Dataset       | Venue     | Supp. Material     |
|---------|---------|---------|---------|---------|---------|---------|
| Sam Shleifer et al |[**Proxy Datasets for Training Convolutional Neural Networks**](https://arxiv.org/pdf/1906.04887.pdf) |  |  Application: Proxy Dataset Generation |  Imagenette, Imagewoof | arXiv, Jun., 2019 |  |
| |[** **]() |  |  Image Classification |  |  | [Code]() |




## 2018
### Papers in 2018 [[Back-to-top](#Dataset-distillation-papers)]

| Author    | Title     | Type      | Task      | Dataset       | Venue     | Supp. Material     |
|---------|---------|---------|---------|---------|---------|---------|
| Tongzhou Wang et al |[**Dataset Distillation**](https://arxiv.org/pdf/1811.10959.pdf) | Bi-level Optimization | Image Classification | MNIST, CIFAR-10 | arXiv, Nov., 2018 | [Code](https://github.com/SsnL/dataset-distillation) |



| |[** **]() |  |  Image Classification |  |  | [Code]() |



