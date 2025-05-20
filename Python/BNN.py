import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


# ------------------------------
# ✅ Dataset
# ------------------------------
# 훈련용
train_transform = transforms.Compose(
    [
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

# 테스트용
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=train_transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# ✅ Binarization Function (STE)
# ------------------------------
class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


binarize = BinaryActivation.apply


# ------------------------------
# ✅ BNN for Training (float weight)
# ------------------------------
class TrainableBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256, bias=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = binarize(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = (x >= 0).float()  # Threshold at 0 (binary output)
        x = self.fc2(x)
        return x


# ------------------------------
# ✅ Verilog-compatible Inference Model
# ------------------------------
class VerilogCompatibleBNN(nn.Module):
    def __init__(self, fc1_weight_bin, fc2_weight_bin, threshold):
        super().__init__()
        self.fc1_weight = fc1_weight_bin
        self.fc2_weight = fc2_weight_bin
        self.threshold = threshold

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = (x >= 0).int()
        xnor1 = 1 - (x.unsqueeze(1) ^ self.fc1_weight.unsqueeze(0))
        pc1 = xnor1.sum(dim=2)
        act1 = (pc1 >= self.threshold).int()
        xnor2 = 1 - (act1.unsqueeze(1) ^ self.fc2_weight.unsqueeze(0))
        pc2 = xnor2.sum(dim=2)
        return pc2


# ------------------------------
# ✅ Training & Evaluation
# ------------------------------
class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super().step(epoch)


def train(model, loader, optimizer, criterion, device):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / total
    acc = 100 * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(acc)
    return acc, avg_loss


def test(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    acc = 100 * correct / total
    val_losses.append(avg_loss)
    val_accuracies.append(acc)
    return acc, avg_loss


def test_verilog_style(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
    acc = correct / len(test_loader.dataset) * 100
    print(f"[Verilog-style 추론] Accuracy: {acc:.2f}%")


# ------------------------------
# ✅ Main
# ------------------------------
model = TrainableBNN().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70)
scheduler = GradualWarmupScheduler(
    optimizer, multiplier=2.0, total_epoch=10, after_scheduler=cosine_scheduler
)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

epochs = 50
early_stop_patience = 8  # 8 epoch 동안 개선 없으면 stop
best_loss = float("inf")
no_improve_count = 0

for epoch in range(1, epochs + 1):
    train_acc, train_loss = train(model, train_loader, optimizer, criterion, device)
    scheduler.step()
    val_acc, val_loss = test(model, test_loader, criterion, device)  # 중간 검증

    print(
        f"[Epoch {epoch}] Train Acc: {train_acc:.2f}%, Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%, Loss: {val_loss:.4f}"
    )

    # Early stopping 조건은 보통 검증 손실(val_loss) 기준으로 함
    if val_loss < best_loss - 1e-4:
        best_loss = val_loss
        no_improve_count = 0
    else:
        no_improve_count += 1

    if no_improve_count >= early_stop_patience:
        print(f"🛑 Early stopping at epoch {epoch+1}")
        break

# 클래스 복사
fc1_w_bin = (model.fc1.weight.data.clone().sign() >= 0).to(device)
fc2_w_bin = (model.fc2.weight.data.clone().sign() >= 0).int().to(device)
thresh = torch.full(
    (256,), 392, dtype=torch.int32, device=device
)  # fixed threshold for now

verilog_model = VerilogCompatibleBNN(fc1_w_bin, fc2_w_bin, thresh).to(device)
test_verilog_style(verilog_model)


# ------------------------------
# ✅ Save Results
# ------------------------------
def save_binary_weights(weight_tensor, filename):
    weight_np = weight_tensor.cpu().numpy()
    bin_weight = (weight_np > 0).astype(np.uint8)
    with open(filename, "w") as f:
        for row in bin_weight:
            f.write("".join(str(b) for b in row) + "\n")


# fc1_weight_bin: shape [256, 784]
# pc_all: shape [10000, 256]
pc_all = []

with torch.no_grad():
    for images, _ in test_loader:
        x = images.view(images.size(0), -1).to(device)
        x_bin = (x >= 0).int()  # [B, 784]
        xnor = 1 - (x_bin.unsqueeze(1) ^ fc1_w_bin.unsqueeze(0))  # [B, 256, 784]
        pc = xnor.sum(dim=2)  # [B, 256]
        pc_all.append(pc)

pc_all = torch.cat(pc_all, dim=0)  # [10000, 256]

# Step 1: median-based threshold (before BN correction)
raw_thresholds = pc_all.median(dim=0).values.float()  # float for computation

# Step 2: BatchNorm correction
bn = model.bn1
gamma = bn.weight.detach().cpu().numpy()  # shape: [256]
beta = bn.bias.detach().cpu().numpy()  # shape: [256]
mean = bn.running_mean.detach().cpu().numpy()
var = bn.running_var.detach().cpu().numpy()
eps = bn.eps

correction = (-beta / (gamma + 1e-5)) * np.sqrt(var + eps)
adjusted_thresholds = (
    raw_thresholds.numpy() + correction
)  # + because we subtract negative shift

# Step 3: Clamp and convert to int
thresholds = np.clip(np.round(adjusted_thresholds), 0, 784).astype(np.int32)
thresholds_tensor = torch.tensor(thresholds, dtype=torch.int32, device=device)


verilog_model = VerilogCompatibleBNN(fc1_w_bin, fc2_w_bin, thresholds_tensor).to(device)
test_verilog_style(verilog_model)

with open("fc1_threshold_bin.txt", "w") as f:
    for i, t in enumerate(thresholds):
        bin_str = format(t, "010b")
        f.write(bin_str + "\n")

save_binary_weights(model.fc1.weight.data.clone().cpu(), "fc1_weight_bin.txt")
save_binary_weights(model.fc2.weight.data.clone().cpu(), "fc2_weight_bin.txt")


plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy Curve")

plt.show()
