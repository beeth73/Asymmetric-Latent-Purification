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


# ==============================================================================
# CELL 4: BULLETPROOF ADVERSARIAL TRAINING (V100 SAFE MODE)
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os
import time

# --- 1. ROCK-SOLID HARDWARE SETTINGS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False    # THE FIX: Disable cuDNN profiling crashes
torch.backends.cudnn.deterministic = True # THE FIX: Force stable algorithms

SAVE_DIR = "Paper2_V100_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. THE MODEL (ResNet-50) ---
classifier = models.resnet50(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 10)
classifier = classifier.to(device)

# --- 3. STABLE PGD-3 ATTACK (PURE FP32, NO AUTOCAST) ---
def pgd_attack_bulletproof(model, x, y, eps=8/255, alpha=3/255, iters=3):
    model.eval() # Freeze BatchNorm stats during attack generation
    
    # Initialize random noise within epsilon ball
    noise = torch.empty_like(x).uniform_(-eps, eps).to(device)
    x_adv = torch.clamp(x + noise, 0, 1).detach()
    
    for _ in range(iters):
        x_adv.requires_grad_(True)
        
        # THE FIX: Ensure memory is perfectly contiguous before cuDNN Conv2d
        x_adv_c = x_adv.contiguous() 
        
        # Pure FP32 Forward Pass (Guarantees no precision underflow)
        output = model(x_adv_c)
        loss = nn.CrossEntropyLoss()(output, y)
        
        # Calculate gradients
        grad = torch.autograd.grad(loss, x_adv_c, retain_graph=False, create_graph=False)[0]
        
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            eta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + eta, min=0, max=1).detach()
            
    return x_adv.detach()

# --- 4. OPTIMIZER & SCHEDULER ---
optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.CrossEntropyLoss()

# --- 5. THE CRASH-PROOF TRAINING LOOP ---
def run_bulletproof_training(epochs=50):
    print(f"🔥[IGNITION] Starting {epochs} Epochs of AT on {device}. Pure FP32 Mode.")
    start_time = time.time()
    
    for epoch in range(epochs):
        classifier.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for data, target in pbar:
            # non_blocking=True speeds up Host-to-Device transfer
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # Generate Attack
            data_adv = pgd_attack_bulletproof(classifier, data, target)
            
            # Train on Attacked Data
            classifier.train() # Re-enable BatchNorm tracking for the actual training step
            optimizer.zero_grad()
            
            # Pure FP32 Forward & Backward
            output = classifier(data_adv.contiguous())
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        
        # Epoch Summary & Auto-Save
        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1:02d}/{epochs} | LR: {scheduler.get_last_lr()[0]:.4f} | Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            path = os.path.join(SAVE_DIR, f"resnet50_SOTA_epoch_{epoch+1}.pth")
            torch.save(classifier.state_dict(), path)
            print(f"   💾[SAVED] {path}")

    final_path = os.path.join(SAVE_DIR, "resnet50_SOTA_FINAL.pth")
    torch.save(classifier.state_dict(), final_path)
    print(f"\n🏆 [MISSION SUCCESS] Final model saved. Total Time: {(time.time()-start_time)/60:.2f} mins")

# LAUNCH
run_bulletproof_training(epochs=50)
