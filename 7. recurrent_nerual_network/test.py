import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import RNN
from config import test_config

def test(config):

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root = config.dataroot, train = False, transform = transform)
    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle = False)

    if config.bidirectional:
        path_index = 'bidirectional/'
    else:
        path_index = 'no_bidirectional/'

    model = RNN(config.input_size, config.hidden_size, config.num_layers,
                config.classes, config.bidirectional).to(device)
    ckpt_path = config.ckpt + '/' + path_index + '5.ckpt'
    model.load_state_dict(torch.load(ckpt_path))

    correct = 0
    total = 0
    for step, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.reshape(-1, config.sequence_length, config.input_size).to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        _, pred = torch.max(outputs.data, dim = 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    configs = test_config()
    test(configs)