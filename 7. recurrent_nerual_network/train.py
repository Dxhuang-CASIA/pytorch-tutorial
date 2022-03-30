import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import RNN
from config import train_config

def train(config):

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root = config.dataroot, train = True,
                                         transform = transform, download = True)
    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle = True)

    model = RNN(config.input_size, config.hidden_size, config.num_layers,
                config.classes, config.bidirectional).to(device)

    if config.bidirectional:
        config.lr = 0.003
        path_index = 'bidirectional/'
    else:
        config.lr = 0.01
        path_index = 'no_bidirectional/'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config.lr)

    loss_list = []
    total_step = len(dataloader)
    for epoch in range(config.epochs):
        model.train()
        for step, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.reshape(-1, config.sequence_length, config.input_size).to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, config.epochs, step + 1, total_step, np.mean(loss_list)))

        loss_list = []
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), config.ckpt + '/' + path_index + str(epoch + 1) + '.ckpt')
            print('Save the model parameter successfully!')

if __name__ == '__main__':
    configs = train_config()
    train(configs)