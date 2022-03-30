import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from config import test_config
from model import model

def test(config):
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root = config.dataroot, train = False, transform = transform)
    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle = False)

    models = model(config.input_size, config.classes)

    print('Loading the weights....')
    models.load_state_dict(torch.load(config.ckpt))

    correct = 0
    total = 0

    for steps, (images, labels) in enumerate(dataloader):
        images = images.reshape(-1, config.input_size)
        outputs = models(images)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    configs = test_config()
    test(configs)