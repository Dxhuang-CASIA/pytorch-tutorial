import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import model
from config import train_config

def train(config):
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root = config.dataroot, train = True,
                                         transform = transform, download = True)
    dataloader = DataLoader(dataset = dataset, batch_size = config.batch_size, shuffle = True)

    models = model(config.input_size, config.classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(models.parameters(), lr = config.lr)

    total_step = len(dataloader)
    for epoch in range(config.epochs):
        for step, (images, labels) in enumerate(dataloader):
            images = images.reshape(-1, config.input_size)

            outputs = models(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, config.epochs, step + 1, total_step, loss.item()))

        if (epoch + 1) % 5 == 0:
            torch.save(models.state_dict(), config.ckpt + '/' + str(epoch + 1) + '.ckpt')
            print('Save the model parameter successfully!')

if __name__ == '__main__':
    config = train_config()
    train(config = config)