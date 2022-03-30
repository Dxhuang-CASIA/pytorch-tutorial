import argparse

def train_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_length', type = int, default = 28)
    parser.add_argument('--input_size', type = int, default = 28)
    parser.add_argument('--hidden_size', type = int, default = 128)
    parser.add_argument('--num_layers', type = int, default = 2)
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 0.01, help = '0.01(n/b) & 0.003(b)')
    parser.add_argument('--classes', type = int, default = 10)
    parser.add_argument('--dataroot', type = str, default = '../data')
    parser.add_argument('--epochs', type = int, default = 5)
    parser.add_argument('--bidirectional', type = bool, default = True)
    parser.add_argument('--ckpt', type = str, default = './ckpt')

    config = parser.parse_args()
    return config

def test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_length', type = int, default = 28)
    parser.add_argument('--input_size', type = int, default = 28)
    parser.add_argument('--hidden_size', type = int, default = 128)
    parser.add_argument('--num_layers', type = int, default = 2)
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--classes', type = int, default = 10)
    parser.add_argument('--dataroot', type = str, default = '../data')
    parser.add_argument('--epochs', type = int, default = 5)
    parser.add_argument('--bidirectional', type = bool, default = False)
    parser.add_argument('--ckpt', type = str, default = './ckpt')

    config = parser.parse_args()
    return config