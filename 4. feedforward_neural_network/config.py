import argparse

def train_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type = int, default = 28 * 28)
    parser.add_argument('--hidden_size', type = int, default = 500)
    parser.add_argument('--classes', type = int, default = 10)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--dataroot', type = str, default = '../data')
    parser.add_argument('--ckpt', type = str, default = './ckpt')

    config = parser.parse_args()

    return config

def test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type = int, default = 28 * 28)
    parser.add_argument('--hidden_size', type = int, default = 500)
    parser.add_argument('--classes', type = int, default = 10)
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--dataroot', type = str, default = '../data')
    parser.add_argument('--ckpt', type = str, default = './ckpt/30.ckpt')

    config = parser.parse_args()

    return config