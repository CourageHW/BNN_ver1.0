import torch
import torchvision
import torchvision.transforms as transforms

# ✅ MNIST 원래 Normalize 기준
mean = 0.1307
std = 0.3081

# ✅ 정규화 포함된 transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# ✅ test 데이터셋 로딩
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)

# ✅ 전체 10000장 가져오기
images, labels = next(iter(test_loader))  # [10000, 1, 28, 28]
images = images.view(-1, 784)             # [10000, 784]

# ✅ 정규화된 값 기준으로 0 이상이면 1, 아니면 0
images_bin = (images >= 0).int().numpy()

# ✅ 저장 (한 줄 = 한 이미지 = 784bit)
with open("mnist_test_bin_images.txt", "w") as f:
    for img_bin in images_bin:
        f.write("".join(str(bit) for bit in img_bin) + "\n")
