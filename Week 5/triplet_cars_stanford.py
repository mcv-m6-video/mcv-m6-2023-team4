import os, sys
import torch
import random
import time
import cv2
import glob
from itertools import combinations

import torch.nn as nn
import torch.optim as optim

from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.autograd import Variable
from tqdm import tqdm
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,precision_score, recall_score, average_precision_score,precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import PrecisionRecallDisplay

from torch.utils.data.sampler import BatchSampler
from pytorch_metric_learning.utils.inference import FaissKNN
import umap.umap_ as umap
#import faiss
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import accuracy_calculator
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image
import torchvision.models as models

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if device.type == "cuda":
    torch.cuda.get_device_name()
    
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
    
        
class ImageEmbeddingNet(nn.Module):
    def __init__(self, hidden_space=1000):
        super(ImageEmbeddingNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.hidden = nn.Linear(2048, hidden_space)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1).squeeze()
        x = self.hidden(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)
    
class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
    
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
gen = torch.Generator()
gen.manual_seed(SEED)

#data_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
data_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
data_transform_train = transforms.Compose([transforms.Resize((224, 224)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

dataset_path = './outputs/dataset/'
train_dataset = torchvision.datasets.ImageFolder(root="../datasets/cars_original/train/", transform=data_transform_train)
val_dataset = torchvision.datasets.ImageFolder(root=dataset_path+"/test/",  transform=data_transform)



# num_samples = len(train_dataset)
# train_size = int(0.1 * num_samples)
# val_size = int(0.1 * num_samples)
# test_size = num_samples - train_size - val_size

# train_dataset, val_dataset, test_coco_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size, test_size], generator=gen)

# print(f'Num images train: {len(train_dataset)}')
# print(f'Num images val: {len(val_dataset)}')
# print(f'Num images test: {len(test_coco_dataset)}')


class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        img3 = Image.open(img3).convert('RGB')
        
        if img1.mode == 'L':
            img1.show()
        elif img2.mode == 'L':
            img2.show()
        elif img3.mode == 'L':
            img3.show()
        
        # if len(img1.shape) > 2:
        #     img1 = Image.fromarray(np.uint8(img1.permute(1, 2, 0).numpy() * 255))
        #     img2 = Image.fromarray(np.uint8(img2.permute(1, 2, 0).numpy() * 255))
        #     img3 = Image.fromarray(np.uint8(img3.permute(1, 2, 0).numpy() * 255))
        # else:
        #     img1 = Image.fromarray(img1.numpy(), mode='L')
        #     img2 = Image.fromarray(img2.numpy(), mode='L')
        #     img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)
    
train_dataset.train = True
val_dataset.train = False
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
gen = torch.Generator()
gen.manual_seed(SEED)

train_dataset.train_data = np.array([s[0] for s in train_dataset.imgs])
val_dataset.test_data = np.array([s[0] for s in val_dataset.imgs])


train_dataset.train_labels = torch.from_numpy(np.array([s[1] for s in train_dataset]))
val_dataset.test_labels = torch.from_numpy(np.array([s[1] for s in val_dataset]))

train_dataset.train = True
triplet_dataset_train = TripletDataset(train_dataset)

val_dataset.train = False
triplet_dataset_val = TripletDataset(val_dataset)

# The space we want to project the embeddings to. This is just a linear transformation
hidden_space_dim = 1000
im_embedding_net = ImageEmbeddingNet(hidden_space_dim)
triplet_net = TripletNet(im_embedding_net)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.get_device_name()
if cuda:
    triplet_net.cuda()
# Set up the network and training parameters
margin = 1.
loss_fn = TripletLoss(margin)
batch_size = 64
lr = 1e-3
optimizer = optim.Adam(triplet_net.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 10
triplet_train_loader = torch.utils.data.DataLoader(triplet_dataset_train, batch_size=batch_size, shuffle=True)
triplet_val_loader = torch.utils.data.DataLoader(triplet_dataset_val, batch_size=batch_size, shuffle=False)

def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()

    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        
        if cuda:
            data = tuple(d.cuda() for d in data)
            
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)
        

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    
    train_losses = []
    val_losses = []
    
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    return train_losses, val_losses

train = True
if train:
    train_losses, val_losses = fit(triplet_train_loader, triplet_val_loader, triplet_net, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    print('Train losses:', train_losses)
    print('Val losses:', val_losses)
    torch.save(triplet_net.state_dict(), 'triplet_nn_pretrained_cars_20epochs_dataaugm.pth')
else:
    triplet_net.load_state_dict(torch.load('triplet_nn_pretrained_cars_20epochs_dataaugm.pth'))