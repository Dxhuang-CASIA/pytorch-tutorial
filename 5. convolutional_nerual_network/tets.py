import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from config import test_config
from model import ConvNet

def test(config):

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root = config.dataroot, train = False, transform = transform)
    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle = False)

    model = ConvNet(config.in_channel, config.classes).to(device)

    print('Loading the weights....')
    model.load_state_dict(torch.load(config.ckpt))

    correct = 0
    total = 0

    model.eval()
    for steps, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    configs = test_config()
    test(configs)