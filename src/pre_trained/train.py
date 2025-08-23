import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from ...models.inception import Inceptionv1WithAux

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
batch_size = 128

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

print("Pre-training on ImageNet-Mini-1000")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

imagenet_train = datasets.ImageFolder(root="./data/train", transform=train_transform)
imagenet_val = datasets.ImageFolder(root="./model/val", transform=val_transform)

train_loader = DataLoader(imagenet_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(imagenet_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

model = Inceptionv1WithAux(in_channels=3, num_classes=1000).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.25, patience=2, verbose=True)

def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(imgs)
        if isinstance(outputs, tuple):
            main_output, aux1, aux2 = outputs
            loss1 = criterion(main_output, labels)
            loss2 = criterion(aux1, labels)
            loss3 = criterion(aux2, labels)
            loss = loss1 + 0.3 * (loss2 + loss3)
            _, preds = main_output.max(1)
        else:
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / total
    acc = 100. * correct / total
    print(f"Epoch {epoch} Train Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    return avg_loss, acc

def evaluate(model, loader, criterion, name="Val"):
    model.eval()
    total_loss, correct_top1, correct_top5, total = 0.0, 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            total += labels.size(0)
            _, pred_top1 = outputs.max(1)
            correct_top1 += pred_top1.eq(labels).sum().item()
            _, pred_top5 = outputs.topk(5, 1, True, True)
            correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()
    top1 = 100. * correct_top1 / total
    top5 = 100. * correct_top5 / total
    avg_loss = total_loss / total
    print(f"{name}: Loss {avg_loss:.4f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")
    return avg_loss, top1, top5

train_losses, train_accs, val_losses, val_accs = [], [], [], []
best_acc = 0

for epoch in range(1, num_epochs + 1):
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, epoch)
    val_loss, val_top1, _ = evaluate(model, val_loader, criterion)
    scheduler.step(val_top1)
    train_losses.append(tr_loss)
    train_accs.append(tr_acc)
    val_losses.append(val_loss)
    val_accs.append(val_top1)
    if val_top1 > best_acc:
        best_acc = val_top1
        torch.save(model.state_dict(), "./model/inceptionv1_imagenet_mini.pth")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("ImageNet-Mini Loss")
plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("ImageNet-Mini Acc")
plt.show()