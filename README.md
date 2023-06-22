# Drop at the Last Moment: Information-Centric Dropout for Privacy-Preserving Neural Network
성균관대학교 2023 2학기 학부 졸업 논문 프로젝트(2023 bachelor graduation thesis)  

| 학번 | 2018310737 
| 이름 |안현준(Hyunjun Ahn)
___
### Environment
```bash
python = 3.7.13
torch = 1.10.1
torchvision = 0.11.2
numpy = 1.21.6
matplotlib = 3.5.2
sklearn = 1.0.2
PIL = 9.1.1
```
___
### Files
```bash
├── metrics
│   ├── distance_celebA.ipynb (Example code to measure ICD(Inter Class Distance) of dataset with custom dataset class)
│   ├── distance.ipynb (Example code to measure ICD(Inter Class Distance) of dataset with ImageFolder)
│   ├── distance_distribution.ipynb (Example code to plot the distribution of ICD(Inter Class Distance) with ImageFolder)
│   ├── distance_distribution.ipynb (Example code to plot the distribution of ICD(Inter Class Distance) with custom dataset class)
│   └── utility.py (To measure utility performance of the trained model (i.e. Accuracy, Precision, Recall, F1-Score))
└── models
    ├── solver (differential dropout modules)
    │   ├── solver.py (Initial version to implement differential dropout)
    │   ├── solver_v2.py (Version without epoch-based score term)
    │   └── solver_v3.py (Final version of differential dropout module)
    ├── train_cifar10.ipynb (Example code to train a model with ImageFolder)
    ├── train_celebA.ipynb (Example code to train a model with custom dataset class)
    ├── ResNet.py (ResNet backbone implementation)
    ├── ViT.py (ViT backbone implementation)
    └── EfficientNet.py (EfficientNet backbone implementation)
```
