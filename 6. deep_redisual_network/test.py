import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import ResNet, ResidualBlock
from config import test_config

def test(config):

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root = config.dataroot, train = False, transform = transform)
    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle = False)

    model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
    model.load_state_dict(torch.load(config.ckpt))

    model.eval()

    correct = 0
    total = 0
    len_data = len(dataloader)
    for step, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        _, pred = torch.max(outputs.data, dim = 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        print('{}/{}'.format(step, len_data))

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    configs = test_config()
    test(configs)