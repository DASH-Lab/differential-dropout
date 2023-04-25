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
batch_size = 256
model = ResNet.resnet50().to(device)
weight_path = './weights/ResNet'
model = torch.load(weight_path)
criterion = nn.CrossEntropyLoss()
data_path = '/media/data1/hyunjun/cifar-10/test/'
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

testset = torchvision.datasets.ImageFolder(root=data_path, transform=trans)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=False,)

classes = testset.classes

def test_loop(dataloader):
    test_loss = 0
    correct = 0

    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += criterion(pred, y.to(device)).item()
            
            pred = torch.argmax(pred, dim=1)
            correct += (pred == y.to(device)).type(torch.float).sum().item()
            
            pred = pred.cpu().numpy()
            y = y.cpu().numpy()
            y_true.extend(y)
            y_pred.extend(pred)
    
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=classes))
    print('Accuracy:', correct / len(dataloader.dataset), 'Loss:', test_loss / len(dataloader.dataset))

test_loop(testloader)
