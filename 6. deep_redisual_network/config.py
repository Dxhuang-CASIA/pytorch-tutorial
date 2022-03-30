import argparse

def train_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type = str, default = '../data')
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--epochs', type = int, default = 80)
    parser.add_argument('--ckpt', type = str, default = './ckpt')

    config = parser.parse_args()
    return config

def test_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', type=str, default='../data')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default='./ckpt/80.ckpt')

    config = parser.parse_args()
    return config