import os
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from cleanlab.filter import find_label_issues
from tqdm import tqdm

# CIFAR-10 경로
BASE_DIR = "../../data/dblab-dataset/uoft_cs_cifar10"

# Transform 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset 불러오기 (train)
trainset = torchvision.datasets.ImageFolder(
    root=BASE_DIR + "/train", transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

# Pretrained 모델 (ResNet18)
model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # CIFAR-10 라벨 개수
model.eval()

# 라벨 예측 (softmax 확률 저장)
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(trainloader):
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        all_probs.append(probs)
        all_labels.append(labels)

all_probs = torch.cat(all_probs).numpy()
all_labels = torch.cat(all_labels).numpy()

# Cleanlab으로 noisy label 탐지
label_issues = find_label_issues(
    labels=all_labels,
    pred_probs=all_probs,
    return_indices_ranked_by="self_confidence"
)

print(f" 탐지된 noisy labels 개수: {len(label_issues)}")

# 저장
pd.DataFrame({"noisy_index": label_issues}).to_csv("cifar10_noisy_labels.csv", index=False)
print(" CIFAR-10 noisy label 결과 저장 완료: cifar10_noisy_labels.csv")
