import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import ResNet, ResidualBlock
from config import train_config

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(config):

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.CIFAR10(root = config.dataroot, train = True,
                                           transform = transform, download = True)
    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle = True)

    model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = config.lr)

    total_step = len(dataloader)
    curr_lr = config.lr

    loss_list = []

    for epoch in range(config.epochs):
        model.train()
        for step, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, config.epochs, step + 1, total_step, np.mean(loss_list)))

        loss_list = []

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), config.ckpt + '/' + str(epoch + 1) + '.ckpt')
            print('Save the model parameter successfully!')

        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

if __name__ == '__main__':
    configs = train_config()
    train(configs)