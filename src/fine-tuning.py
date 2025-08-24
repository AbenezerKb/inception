import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from ..models.inception import Inceptionv1WithAux
from pre_trained.train import train_epoch, evaluate

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
num_epochs = 20
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fine_tune_single_label(dataset_name, num_classes, root, download=False):
    print(f"\nFine-tuning on {dataset_name}")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4) if "CIFAR" in dataset_name else transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    if dataset_name == "CIFAR-10":
        train_ds = datasets.CIFAR10(root=root, train=True, download=download, transform=transform_train)
        test_ds = datasets.CIFAR10(root=root, train=False, download=download, transform=transform_test)
    elif dataset_name == "CIFAR-100":
        train_ds = datasets.CIFAR100(root=root, train=True, download=download, transform=transform_train)
        test_ds = datasets.CIFAR100(root=root, train=False, download=download, transform=transform_test)
    elif dataset_name == "LSUN":
        train_ds = datasets.LSUN(root=root, classes='train', transform=transform_train)
        test_ds = datasets.LSUN(root=root, classes='test', transform=transform_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = Inceptionv1WithAux(in_channels=3, num_classes=1000).to(device)
    model.load_state_dict(torch.load("./model/inceptionv1_imagenet_mini.pth"))
    
    model.fc1 = nn.Linear(model.fc1.in_features, num_classes).to(device)
    model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes).to(device)
    model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.25, patience=2, verbose=True)
    
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    best_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, epoch)
        te_loss, te_top1, _ = evaluate(model, test_loader, criterion, name=f"{dataset_name} Test")
        scheduler.step(te_top1)
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_losses.append(te_loss)
        test_accs.append(te_top1)
        if te_top1 > best_acc:
            best_acc = te_top1
            torch.save(model.state_dict(), f"/kaggle/working/inceptionv1_{dataset_name.lower().replace('-', '')}.pth")
        
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.title(f"{dataset_name} Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(test_accs, label="Test Acc")
    plt.legend()
    plt.title(f"{dataset_name} Acc")
    plt.show()


fine_tune_single_label("CIFAR-10", 10, root="./model/cifar-10", download=False)


fine_tune_single_label("CIFAR-100", 100, root="./model/cifar-100-python", download=False)


fine_tune_single_label("LSUN", 10, root="/kaggle/input/lsun", download=False)