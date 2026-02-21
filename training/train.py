import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

# Modern CNN Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 48, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        
        # Conv Block 2
        self.conv3 = nn.Conv2d(96, 144, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(144, 192, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Fully Connected
        # After two pools: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(192 * 7 * 7, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Block 2
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.flatten(x, 1)
        x = F.gelu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Label smoothing for robustness
        loss = F.cross_entropy(output, target, label_smoothing=0.1)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if batch_idx % 200 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    # Optimize for M1 Pro (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Training on: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training on: CUDA")
    else:
        device = torch.device("cpu")
        print("Training on: CPU")
    
    # Modern Data Augmentation for Robustness
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = Net().to(device)
    
    epochs = 15
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # OneCycleLR is modern gold standard for fast convergence
    scheduler = OneCycleLR(optimizer, max_lr=1e-2, 
                          steps_per_epoch=len(train_loader), 
                          epochs=epochs)

    best_acc = 0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, scheduler, epoch)
        acc = test(model, device, test_loader)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "mnist_cnn.pt")
            print(f"Found new best model with {acc:.2f}% accuracy! Saved.")

    print(f"Final Best Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
