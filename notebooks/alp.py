# Cell 1
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import time

# Gate 9: Determinism for Independent Replication
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Hardware Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ [HARDWARE] GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ [VRAM] Total Capacity: {vram:.2f} GB")
else:
    print("❌ [ERROR] CUDA not detected. Check your drivers!")

# Create Directory for Paper 2 Results
SAVE_DIR = "Paper2_V100_Results"
os.makedirs(SAVE_DIR, exist_ok=True)


# Cell 2

# SOTA Data Augmentation (No shortcuts for IEEE!)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Large Batch Size (256) is safe on 32GB VRAM
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

print(f"📊 [DATA] CIFAR-10 Loaded. Batches: {len(train_loader)} (Train), {len(test_loader)} (Test)")


# cell 3

from torchvision.models import resnet50

def get_resnet50_victim():
    # Load model (Uncomment weights=None if you want a true scratch baseline)
    model = resnet50(weights=None) 
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)

classifier = get_resnet50_victim()
print(f"🧠 [MODEL] ResNet-50 initialized. Parameters: {sum(p.numel() for p in classifier.parameters()):,}")


# Cell 4a

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import time
import gc

# 1. THE NUCLEAR OPTION: Disable the features that are causing the driver to crash
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True # We leave it on for training, but off for the attack

# 2. VRAM PURGE
torch.cuda.empty_cache()
gc.collect()

DEVICE = torch.device("cuda")
SAVE_DIR = "./Paper2_V100_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"🛰️ [SYSTEM] Hardware Stabilized. Batch Size Target: 32. Mode: Atomic Safety.")

# cell 4b
# 1. THE MODEL
classifier = models.resnet50(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 10)
classifier = classifier.to(DEVICE)

# 2. OPTIMIZATION
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

print("✅ [SUCCESS] Model and Optimizer Ready.")

#cell 4c
def pgd_attack_atomic(model, x, y, eps=8/255, alpha=3/255, iters=3):
    model.eval()
    
    # We move the image to a 'detached' state to keep the graph simple
    x_adv = x.clone().detach() + torch.zeros_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    # Howard: "We use a context manager to ensure cuDNN doesn't try anything clever"
    with torch.backends.cudnn.flags(enabled=False): # Bypasses the 'FIND engine' error!
        for _ in range(iters):
            x_adv.requires_grad = True
            output = model(x_adv)
            loss = criterion(output, y)
            
            grad = torch.autograd.grad(loss, x_adv)[0]
            
            with torch.no_grad():
                x_adv = x_adv + alpha * grad.sign()
                eta = torch.clamp(x_adv - x, min=-eps, max=eps)
                x_adv = torch.clamp(x + eta, min=0, max=1).detach()
                
    return x_adv

print("🛡️ [SUCCESS] Atomic Attacker Ready. cuDNN-Bypass active.")

#cell 4D
# Rebuild loader with Batch Size 32 for maximum VRAM safety
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=32, shuffle=True, num_workers=2, drop_last=True
)

print(f"🔥 [IGNITION] Starting 20-Epoch Sprint. Batch Size: 32.")

for epoch in range(1, 21):
    classifier.train()
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/20]")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 1. Generate Attack using the Atomic (cuDNN-bypass) method
        adv_images = pgd_attack_atomic(classifier, images, labels)

        # 2. Train (cuDNN enabled here for speed, batch size 32 is safe)
        classifier.train()
        optimizer.zero_grad()
        outputs = classifier(adv_images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    print(f"✅ Epoch {epoch} complete. Avg Loss: {epoch_loss/len(train_loader):.4f}")

    if epoch % 5 == 0:
        torch.save(classifier.state_dict(), os.path.join(SAVE_DIR, f"resnet50_V100_final.pth"))

print("🏆 MISSION COMPLETE.")
