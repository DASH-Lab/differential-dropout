import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
from ..models import DenseNet, EfficientNet, MobileNetV2, ResNet, ViT

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##########################################
######### Configure Metric Setting ####### 
##########################################
batch_size = 1
model = ResNet.resnet50().to(device)
weight_path = './weights/ResNet'
model = torch.load(weight_path)
criterion = nn.CrossEntropyLoss()
member_path = '/media/data1/hyunjun/cifar-10/train/'
nonmember_path = '/media/data1/hyunjun/cifar-10/test/'
data = 'ImageNet'
##########################################
##########################################

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) 
if data == 'STL-10':
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
elif data == 'Cifar-10':
    mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
elif data == 'CelebA':
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

nonmember_set = torchvision.datasets.ImageFolder(root=nonmember_path, transform=trans)
nonmember_loader = DataLoader(nonmember_set, batch_size=batch_size, shuffle=True, drop_last=False,)
member_set = torchvision.datasets.ImageFolder(root=member_path, transform=trans)
member_loader = DataLoader(member_set, batch_size=batch_size, shuffle=True, drop_last=False,)

classes = nonmember_set.classes

def random_zeroing(X, factor):
    origin = X.numpy()
    for i in range(len(origin)):
        for j in range(len(origin[i])):
            mask = np.random.choice([1.0, 0.0], origin[i][j].shape, p=[1-factor, factor])
            origin[i][j] = origin[i][j] * mask
    
    return torch.Tensor(origin)

def label_resistance_loop(dataloader):
    model.eval()
    
    perturbation = []
    
    with torch.no_grad():
        for X, y in dataloader:
            perturbation_factor = 0.0
            x = random_zeroing(X, perturbation_factor)
            initial_pred = None
            while(True):
                pred = model(x.to(device))
                pred = torch.argmax(pred, dim=1)
                pred = pred.cpu().numpy()
                if initial_pred is None:
                    initial_pred = pred[0]
                elif initial_pred != pred[0]:
                    perturbation.append(perturbation_factor)
                    break
                else:
                    perturbation_factor += 0.05
                if perturbation_factor > 1.0:
                    break
                x = random_zeroing(X, perturbation_factor)
                    
    return np.array(perturbation)

print("#################################")
print("## Non-member Label resistance ##")
print("#################################")
non_memeber = label_resistance_loop(nonmember_loader)

print("#################################")
print("#### member Label resistance ####")
print("#################################")
memeber = label_resistance_loop(member_loader)

print(np.mean(non_memeber), np.var(non_memeber))

uniques, count = np.unique(non_memeber, return_counts=True)
count = count / np.sum(count)
plt.plot(uniques, count)
print(np.mean(memeber), np.var(memeber))

uniques, count = np.unique(memeber, return_counts=True)
count = count / np.sum(count)
plt.plot(uniques, count)
plt.show()