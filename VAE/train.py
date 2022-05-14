import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from typing import List

from model import VAE


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
train_dataset = datasets.MNIST(
    "../data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST("../data", train=False, transform=transform)

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 10
KLD_WEIGHT = 0.01


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    cum_rec_loss = 0
    cum_kl_loss = 0
    cum_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        results = model(data)
        train_loss = model.loss_function(*results, KLD_WEIGHT)
        #         print(epoch, {key: val.item() for key, val in train_loss.items()})
        cum_loss += train_loss["loss"].item()
        cum_rec_loss += train_loss["Reconstruction_Loss"].item()
        cum_kl_loss += train_loss["KLD"].item()
        train_loss["loss"].backward()
        optimizer.step()
    print(
        f"train - epoch = {epoch}, loss = {cum_loss / len(train_loader)}, rec_loss = {cum_rec_loss / len(train_loader)}, kl_loss = {cum_kl_loss / len(train_loader)}"
    )


def test(model, device, test_loader):
    model.eval()
    cum_rec_loss = 0
    cum_kl_loss = 0
    cum_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            results = model(data)
            test_loss = model.loss_function(*results, KLD_WEIGHT)
            cum_loss += test_loss["loss"].item()
            cum_rec_loss += test_loss["Reconstruction_Loss"].item()
            cum_kl_loss += test_loss["KLD"].item()
    print(
        f"validation - loss = {cum_loss / len(test_loader)}, rec_loss = {cum_rec_loss / len(test_loader)}, kl_loss = {cum_kl_loss / len(test_loader)}"
    )


model = VAE(1, 28).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

    torch.save(model, "./model.pth")
