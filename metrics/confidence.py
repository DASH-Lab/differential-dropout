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
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.Resize((224, 224)),
])

nonmember_set = torchvision.datasets.ImageFolder(root=nonmember_path, transform=trans)
nonmember_loader = DataLoader(nonmember_set, batch_size=batch_size, shuffle=True, drop_last=False,)
member_set = torchvision.datasets.ImageFolder(root=member_path, transform=trans)
member_loader = DataLoader(member_set, batch_size=batch_size, shuffle=True, drop_last=False,)

classes = nonmember_set.classes

def softmax_entropy(pred):
    score = pred
    score = np.exp(score) / np.sum(np.exp(score))
    score = score * (0.0 - np.log2(score))
    return np.sum(score)

def confidence_measure_loop(dataloader):
    model.eval()
    
    conf_entropy = []
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy()
            for i in range(len(pred)):
                conf_entropy.append(softmax_entropy(pred[i]))
            
    return np.array(conf_entropy)

print("#################################")
print("## Non-member Label resistance ##")
print("#################################")
non_memeber = confidence_measure_loop(nonmember_loader)

print("#################################")
print("#### member Label resistance ####")
print("#################################")
memeber = confidence_measure_loop(member_loader)