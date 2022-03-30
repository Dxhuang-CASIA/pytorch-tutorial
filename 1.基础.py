import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

######## Basic autograd example 1 ########
x = torch.tensor(1., requires_grad = True)
w = torch.tensor(2., requires_grad = True)
b = torch.tensor(3., requires_grad = True)

y = w * x + b

y.backward()

print(x.grad) # 对x求导
print(w.grad) # 对w求导
print(b.grad) # 对b求导

######## Basic autograd example 2 ########
x = torch.randn(10, 3)
y = torch.randn(10, 2)

linear = nn.Linear(3, 2)
print('w:', linear.weight)
print('b:', linear.bias)

criterion = nn.MSELoss()
optimizer = optim.SGD(linear.parameters(), lr = 0.01)

pred = linear(x)

loss = criterion(pred, y)
print('loss:', loss.item())

loss.backward()

print ('dL/dw:', linear.weight.grad)
print ('dL/db:', linear.bias.grad)

optimizer.step()

pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization:', loss.item())

######## Loading data from numpy ########
x = np.array([[1, 2], [3, 4]])

y = torch.from_numpy(x)

z = y.numpy()

######## Input pipline ########
train_dataset = torchvision.datasets.CIFAR10(root = './data/', train = True,
                                             transform = transforms.ToTensor(), download = True)

image, label = train_dataset[0]
print(image.size())
print(label)

train_dataloader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)

data_iter = iter(train_dataloader)

for images, labels in train_dataloader:
    pass

######## Input pipline for custom dataset ########
class CustomDataset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index):
        pass
    def __len__(self):
        return 0

custom_dataset = CustomDataset()
train_loader = DataLoader(custom_dataset, batch_size = 64, shuffle = True)

######## Pretrained model ########
resnet_pretrain = torchvision.models.resnet18(pretrained = True)

# finetune only the top layer of the model
for param in resnet_pretrain.parameters():
    param.requires_grad = False

# replace the top layer for finetuning
resnet_pretrain.fc = nn.Linear(resnet_pretrain.fc.in_features, 100) # 变成100分类

images = torch.randn(64, 3, 224, 224)
outputs = resnet_pretrain(image)
print(outputs.size()) # [64, 100]

######## Save and load the model ########

# Save and load the entire model.
torch.save(resnet_pretrain, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet_pretrain.state_dict(), 'params.ckpt')
resnet_pretrain.load_state_dict(torch.load('params.ckpt'))