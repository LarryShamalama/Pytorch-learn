from __future__ import print_function

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

NUM_EPOCHS = 20

cuda_available = torch.cuda.is_available()
print('CUDA is {}available'.format('not '*(not cuda_available)))

mnist_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./mnist_data', train=True, transform=mnist_transforms, download=True)
mnist_test  = torchvision.datasets.MNIST(root='./mnist_data', train=False, transform=mnist_transforms, download=True)

trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True, num_workers=2)


class Classifier(nn.Module):
    """Convnet Classifier"""
    def __init__(self):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # Logistic Regression
        self.clf = nn.Linear(128, 10)

    def forward(self, x):
        return self.clf(self.conv(x).squeeze())
    
    
def to_one_hot(array, num_classes):
    '''
    for labels
    '''
    n = len(array)
    one_hot = torch.zeros(n, num_classes)
    
    for i, j in enumerate(array):
        one_hot[i, j] = 1
    
    return one_hot



if __name__ == "__main__":
    clf = Classifier()
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        
        start = time.time()
        
        total_train = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            
            optimizer.zero_grad() # clears all gradients
            #targets = to_one_hot(targets, 10)
            
            outputs = clf(inputs)
            loss    = criterion(outputs, targets)
            # dim(outputs) = n x 10
            # dim(targets) = n
            
            loss.backward()
            optimizer.step()

            total_train += targets.size(0)
            
        print(total_train)
            
        total   = 0
        correct = 0
        
        end = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            #inputs, targets = Variable(inputs), Variable(targets)
            outputs = clf(inputs)
            
            _, predicted = torch.max(outputs, 1)
            
            total   += targets.size(0)
            correct += float(sum(predicted == targets))
            
        print('Epoch %d: %.3f accuracy, time to train: %.2fs' % (epoch, correct/total, end - start))
