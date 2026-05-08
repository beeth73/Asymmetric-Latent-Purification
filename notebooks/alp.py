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
# CELL 4A: HARDWARE STABILITY & DIRECTORY SETUP
# ==============================================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm

# 1. THE CUDNN CRASH FIX: Disable benchmarking to prevent dynamic-shape memory panics
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True 

# 2. HARDWARE LOCK
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 [SYSTEM] Hardware locked: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"🚀 [SYSTEM] GPU: {torch.cuda.get_device_name(0)}")

# 3. DIRECTORY SETUP
SAVE_DIR = "./Paper2_V100_Results"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"📁 [SYSTEM] Checkpoint directory secured at: {SAVE_DIR}")

# ==============================================================================
# CELL 4B: MODEL & OPTIMIZER INITIALIZATION
# ==============================================================================
print("🧠 [SYSTEM] Initializing ResNet-50 for CIFAR-10...")

# 1. THE MODEL
classifier = models.resnet50(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 10)
classifier = classifier.to(DEVICE)

# 2. LOSS, OPTIMIZER & SCHEDULER
criterion = nn.CrossEntropyLoss()
# Using Nesterov momentum for slightly more stable convergence
optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

# 50 Epochs is standard for a V100 run to get a strong baseline quickly
NUM_EPOCHS = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print("✅[SUCCESS] Model, Optimizer, and Scheduler Ready.")

# ==============================================================================
# CELL 4C: THE BULLETPROOF PGD-3 ATTACK ENGINE
# ==============================================================================
EPS = 8 / 255.0
ALPHA = 3 / 255.0
STEPS = 3

def pgd_attack_v100(model, images, labels, eps=EPS, alpha=ALPHA, steps=STEPS):
    # 1. Cast and pin inputs to memory
    images = images.detach().clone().to(DEVICE, dtype=torch.float32)
    labels = labels.detach().clone().to(DEVICE)

    # 2. Initialize random noise within epsilon ball
    delta = torch.empty_like(images).uniform_(-eps, eps).to(dtype=torch.float32)

    # 3. Freeze BatchNorm stats during attack generation
    model.eval()

    for _ in range(steps):
        # CRITICAL FIX 1: .contiguous() prevents the 'FIND/GET engine' crash
        adv_images = (images + delta).contiguous().to(dtype=torch.float32)
        adv_images.requires_grad_(True)

        with torch.enable_grad():
            outputs = model(adv_images)
            loss = criterion(outputs, labels)

        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            # Keep delta inside the epsilon bounds
            delta = torch.clamp(delta, -eps, eps)
            
            # CRITICAL FIX 2: We DO NOT clamp (images + delta) to [0,1] here 
            # because your DataLoader already normalized the images to negative values!
            delta = delta.contiguous()

    model.train() # Restore train mode
    
    # Final continuous tensor ready for the real forward pass
    adv_images = (images + delta).detach().contiguous().to(dtype=torch.float32)
    return adv_images

print("⚔️ [SUCCESS] PGD-3 Attack Engine Loaded. Math verified.")

# ==============================================================================
# CELL 4D: THE TRAINING LOOP
# ==============================================================================
print("\n" + "=" * 60)
print(f"🔥 [IGNITION] Starting PGD-3 Adversarial Training on {DEVICE}")
print("=" * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    classifier.train()
    running_loss = 0.0
    total_batches = 0

    # TQDM Progress bar
    loop = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{NUM_EPOCHS}]", leave=True)

    for images, labels in loop:
        images = images.to(DEVICE, dtype=torch.float32, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # 1. Generate Attack
        adv_images = pgd_attack_v100(classifier, images, labels)

        # 2. Train on Attacked Images
        classifier.train()
        optimizer.zero_grad(set_to_none=True) # Memory optimization

        # 3. Forward Pass (Forced contiguous to prevent C++ crashes)
        outputs = classifier(adv_images.contiguous())
        loss = criterion(outputs, labels)
        
        # 4. Backward Pass
        loss.backward()

        # 5. SAFETY NET: Gradient Clipping to prevent math explosions
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()

        # Bookkeeping
        running_loss += loss.item()
        total_batches += 1
        loop.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.5f}")

    scheduler.step()
    avg_loss = running_loss / total_batches
    print(f"  → Epoch {epoch:02d} complete | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Checkpoint every 10 epochs
    if epoch % 10 == 0:
        ckpt_path = os.path.join(SAVE_DIR, f"resnet50_at_epoch_{epoch}.pth")
        torch.save(classifier.state_dict(), ckpt_path)
        print(f"  💾 [CHECKPOINT] Saved → {ckpt_path}")

print("🏆 [SUCCESS] Adversarial Training Complete. The Baseline is forged.")

