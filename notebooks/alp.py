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


#cell4

from torch.cuda.amp import autocast, GradScaler

# 1. PGD Attacker for Training (Fast 7-step)
def pgd_attack_train(model, x, y, eps=8/255, alpha=2/255, iters=7):
    model.eval()
    x_adv = x.clone().detach() + torch.FloatTensor(x.shape).uniform_(-eps, eps).to(device)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    for _ in range(iters):
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

# 2. Optimization Setup
optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scaler = GradScaler() # For Mixed Precision Speed
criterion = nn.CrossEntropyLoss()

# 3. The Master Training Loop
def run_adversarial_training(epochs=100):
    print(f"🔥 [IGNITION] Starting 100 Epochs of Adversarial Training on {torch.cuda.get_device_name(0)}")
    
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/100")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            # PHASE A: Generate Adversarial Examples (The Training Attack)
            with torch.no_grad():
                adv_imgs = pgd_attack_train(classifier, imgs, lbls)
            
            # PHASE B: Train on Adversarial Examples
            classifier.train()
            optimizer.zero_grad()
            
            with autocast(): # Mixed Precision Magic
                outputs = classifier(adv_imgs)
                loss = criterion(outputs, lbls)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.4f}"})
        
        scheduler.step()

        # GATE 2 CHECK: Save Checkpoint every 10 Epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"resnet50_at_epoch_{epoch+1}.pth")
            torch.save(classifier.state_dict(), checkpoint_path)
            print(f"💾 [GATE 2] Checkpoint saved: {checkpoint_path}")

    # Final Save
    torch.save(classifier.state_dict(), os.path.join(SAVE_DIR, "resnet50_at_final.pth"))
    print("🏆 [PHASE 1 COMPLETE] ResNet-50 is now Battle-Hardened.")

# LAUNCH
run_adversarial_training()
