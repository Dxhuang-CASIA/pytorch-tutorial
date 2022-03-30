import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import train_config
from model import NerualNet

def train(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root = config.dataroot, train = True,
                                         transform = transform, download = True)
    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle = True)

    model = NerualNet(config.input_size, config.hidden_size, config.classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config.lr)

    total_step = len(dataloader)
    for epoch in range(config.epochs):
        for step, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.reshape(-1, config.input_size).to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, config.epochs, step + 1, total_step, loss.item()))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), config.ckpt + '/' + str(epoch + 1) + '.ckpt')
            print('Save the model parameter successfully!')


if __name__ == '__main__':
    configs = train_config()
    train(configs)