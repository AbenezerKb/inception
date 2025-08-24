import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from models.inception import Inceptionv1WithAux

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20
batch_size = 128
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

print("\nFine-tuning on Pascal VOC 2012 (Multi-Label)")

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
num_voc_classes = len(voc_classes)
voc_class_idx = {cls: i for i, cls in enumerate(voc_classes)}

class VOCClassification(Dataset):
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None):
        self.voc = datasets.VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self.transform = transform
    
    def __len__(self):
        return len(self.voc)
    
    def __getitem__(self, idx):
        img, target = self.voc[idx]
        if self.transform:
            img = self.transform(img)
        
        
        objs = target['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]
        labels = torch.zeros(num_voc_classes)
        for obj in objs:
            cls_name = obj['name']
            if cls_name in voc_class_idx:
                labels[voc_class_idx[cls_name]] = 1
        return img, labels

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
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

voc_train = VOCClassification(root="/kaggle/input/pascal-voc-2012", image_set='trainval', download=False, transform=transform_train)
voc_test = VOCClassification(root="/kaggle/input/pascal-voc-2012", image_set='val', download=False, transform=transform_test)

voc_train_loader = DataLoader(voc_train, batch_size=batch_size, shuffle=True, num_workers=2)
voc_test_loader = DataLoader(voc_test, batch_size=batch_size, shuffle=False, num_workers=2)

model = Inceptionv1WithAux(in_channels=3, num_classes=1000).to(device)
model.load_state_dict(torch.load("./model/inceptionv1_imagenet_mini.pth"))

model.fc1 = nn.Linear(model.fc1.in_features, num_voc_classes).to(device)
model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_voc_classes).to(device)
model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_voc_classes).to(device)

criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=2, verbose=True)

def train_epoch_ml(model, loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        
        outputs = model(imgs)
        if isinstance(outputs, tuple):
            main_output, aux1, aux2 = outputs
            loss1 = criterion(main_output, labels)
            loss2 = criterion(aux1, labels)
            loss3 = criterion(aux2, labels)
            loss = loss1 + 0.3 * (loss2 + loss3)
        else:
            loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    avg_loss = running_loss / total
    print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_ml(model, loader, criterion, name="Test"):
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device).float()
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
    avg_loss = total_loss / total
    print(f"{name}: Loss {avg_loss:.4f}")
    return avg_loss

train_losses, test_losses = [], []
best_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    tr_loss = train_epoch_ml(model, voc_train_loader, optimizer, criterion, epoch)
    te_loss = evaluate_ml(model, voc_test_loader, criterion, name="VOC Test")
    scheduler.step(te_loss)
    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    if te_loss < best_loss:
        best_loss = te_loss
        torch.save(model.state_dict(), "./model/inceptionv1_voc.pth")

plt.figure(figsize=(6, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("VOC 2012 Loss")
plt.show()

print("Training completed successfully!")