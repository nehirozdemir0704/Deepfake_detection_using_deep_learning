import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# 1) Cihaz seçimi (GPU varsa GPU, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2) Veri yolları
# Deepfake_detection_using_deep_learning-master/
#   data/
#     my_dataset/
#       real/
#       fake/
data_dir = os.path.join("..", "data", "my_dataset")

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Veri klasörü bulunamadı: {data_dir}")

# 3) Transformlar (resimleri ortak boyuta getir + normalize et)
img_size = 224
data_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 4) ImageFolder ile dataset oluştur
# Klasör isimleri sınıf ismi: real, fake
full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
class_names = full_dataset.classes
print("Sınıflar:", class_names)
print("Toplam görüntü sayısı:", len(full_dataset))

if len(full_dataset) < 10:
    raise ValueError("Veri seti çok küçük, klasörleri kontrol et (real/fake).")

# 5) Train/validation ayrımı (örneğin %80 train, %20 val)
val_ratio = 0.2
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 6) Model: ResNet18 (pretrained) + son katmanı 2 sınıf için değiştir
model = models.resnet18(weights=None)  # veya pretrained=False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 sınıf: real, fake
model = model.to(device)

# 7) Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 8) Eğitim döngüsü
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size

    print(f"Epoch {epoch} TRAIN - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")


def eval_one_epoch(epoch):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / val_size
    epoch_acc = running_corrects.double() / val_size

    print(f"Epoch {epoch} VAL   - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")


num_epochs = 3  # İstersen arttırabilirsin

for epoch in range(1, num_epochs + 1):
    train_one_epoch(epoch)
    eval_one_epoch(epoch)

# 9) Modeli kaydet
os.makedirs("saved_models", exist_ok=True)
model_path = os.path.join("saved_models", "deepfake_frame_resnet18.pth")
torch.save(model.state_dict(), model_path)
print("Model kaydedildi:", model_path)
