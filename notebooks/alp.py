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


# Cell 4

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm # FIXED: Correct function import
import os
import time

# 1. HARDWARE & OPTIMIZATION SETTINGS
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True # Howard: "The Speed Booster!"
SAVE_DIR = "Paper2_V100_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. DATA LOADERS (Optimized for V100 32GB)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Jeff Dean: "Increasing batch_size to 512 to saturate the V100s cores"
train_loader = DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform_train), 
    batch_size=512, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
)

# 3. THE MODEL
classifier = models.resnet50(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 10)
classifier = classifier.to(device)

# 4. SOTA OPTIMIZATION (2026 Standards)
optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda') # FIXED: 2026 syntax

# 5. OPTIMIZED ATTACK (3-step PGD for training speed)
def pgd_attack_fast(model, x, y, eps=8/255, alpha=3/255, iters=3):
    model.eval()
    x_adv = x.clone().detach() + torch.zeros_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    for _ in range(iters):
        x_adv.requires_grad = True
        with torch.amp.autocast('cuda'): # Faster
            output = model(x_adv)
            loss = criterion(output, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

# 6. THE HEARTBEAT TRAINING LOOP
print(f"🔥 [SUPERSONIC] Starting 50 Epochs. Target: 4:00 PM. VRAM: 32GB Ready.")

def run_mission_critical_training(epochs=50):
    start_time = time.time()
    for epoch in range(epochs):
        classifier.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Fast PGD-3 for training
            data_adv = pgd_attack_fast(classifier, data, target)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                output = classifier(data_adv)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        
        # 💾 AUTO-SAVE every 10 epochs (Speed compromise)
        if (epoch + 1) % 10 == 0:
            path = os.path.join(SAVE_DIR, f"resnet50_SOTA_epoch_{epoch+1}.pth")
            torch.save(classifier.state_dict(), path)
            print(f"\n✅ [SECURED] {path}")

    # FINAL EXPORT
    final_path = os.path.join(SAVE_DIR, "resnet50_SOTA_FINAL.pth")
    torch.save(classifier.state_dict(), final_path)
    print(f"🏆 [MISSION ACCOMPLISHED] Model saved at {final_path}")
    print(f"⏱️ Total Training Time: {(time.time()-start_time)/60:.2f} mins")

# EXECUTE
run_mission_critical_training()

