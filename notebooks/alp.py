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


# =============================================================================
# CELL 4 — Bulletproof PGD-3 Adversarial Training Loop (V100 / cuDNN stable)
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm

# ── 1. cuDNN STABILITY FLAGS ──────────────────────────────────────────────────
# benchmark=True triggers auto-tuner which can fail on dynamic-shaped tensors
# inside the PGD loop; False forces cuDNN to use a deterministic algorithm
# selection path that is always available.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True   # reproducibility bonus
torch.backends.cudnn.enabled = True         # keep cuDNN on — just no benchmark

# Completely disable AMP / autocast. FP16 underflow is a known silent killer
# for small PGD perturbation magnitudes (8/255 ≈ 0.031). Pure FP32 only.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device : {DEVICE}")
print(f"[INFO] cuDNN version : {torch.backends.cudnn.version()}")

# ── 2. OUTPUT DIRECTORY ───────────────────────────────────────────────────────
SAVE_DIR = "./Paper2_V100_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 3. MODEL ──────────────────────────────────────────────────────────────────
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)   # CIFAR-10 → 10 classes
model = model.to(DEVICE)
model.train()

# ── 4. LOSS / OPTIMIZER / SCHEDULER ──────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,     # Nesterov often converges faster; drop if unwanted
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# ── 5. PGD-3 ATTACK ───────────────────────────────────────────────────────────
EPS   = 8  / 255.0   # ℓ∞ radius
ALPHA = 3  / 255.0   # step size
STEPS = 3            # iterations

def pgd_attack(model, images, labels, eps=EPS, alpha=ALPHA, steps=STEPS):
    """
    PGD-ℓ∞ attack, hardened for cuDNN stability:

    • All tensors are explicitly cast to float32 before any conv operation.
    • Every tensor that enters the forward pass is made contiguous so cuDNN
      never receives a strided / non-contiguous view.
    • Gradient computation is isolated with torch.no_grad() outside the
      inner loop and torch.enable_grad() inside — the model stays in eval()
      mode for the duration so BatchNorm running-stats are not corrupted.
    • Cloning + detaching means the attack graph never bleeds into the
      clean training graph.
    """

    # ── Cast & pin inputs ──────────────────────────────────────────────────
    images = images.detach().clone().to(DEVICE, dtype=torch.float32)
    labels = labels.detach().clone().to(DEVICE)

    # ── Random uniform init inside the epsilon ball ────────────────────────
    delta = torch.empty_like(images).uniform_(-eps, eps)
    delta = delta.to(dtype=torch.float32)

    # Switch to eval so BN uses running stats, not batch stats, during attack
    model.eval()

    for _ in range(steps):
        # Strict contiguity check before every forward pass
        adv_images = (images + delta).contiguous().to(dtype=torch.float32)

        adv_images.requires_grad_(True)

        # Forward pass under enable_grad (model is in eval, no autocast)
        with torch.enable_grad():
            outputs = model(adv_images)
            loss    = criterion(outputs, labels)

        # Backward — only w.r.t. adv_images, not model params
        grad = torch.autograd.grad(
            loss,
            adv_images,
            retain_graph=False,
            create_graph=False,
        )[0]

        # Signed gradient step
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            # Project back into ℓ∞ ball
            delta = torch.clamp(delta, -eps, eps)
            # Keep adversarial image in valid pixel range [0, 1]
            # (assuming inputs are already normalised to [0,1])
            delta = torch.clamp(images + delta, 0.0, 1.0) - images
            # Guarantee contiguity for the next iteration
            delta = delta.contiguous()

    model.train()   # restore train mode

    adv_images = (images + delta).detach().contiguous().to(dtype=torch.float32)
    return adv_images


# ── 6. TRAINING LOOP ─────────────────────────────────────────────────────────
NUM_EPOCHS = 50

print("\n" + "=" * 65)
print("  Starting Adversarial Training — PGD-3 | ResNet-50 | CIFAR-10")
print("=" * 65 + "\n")

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()
    running_loss = 0.0
    total_batches = 0

    loop = tqdm(
        train_loader,
        desc=f"Epoch [{epoch:02d}/{NUM_EPOCHS}]",
        leave=True,
        dynamic_ncols=True,
    )

    for batch_idx, (images, labels) in enumerate(loop):
        # Move to device in FP32 explicitly
        images = images.to(DEVICE, dtype=torch.float32, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # ── Generate adversarial examples (model switched to eval inside) ──
        adv_images = pgd_attack(model, images, labels)

        # ── Standard training step on adversarial examples ────────────────
        model.train()
        optimizer.zero_grad(set_to_none=True)  # slightly faster than False

        # Contiguity guard before the "real" forward pass too
        adv_images = adv_images.contiguous()
        outputs    = model(adv_images)
        loss       = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping — prevents occasional exploding-grad spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ── Bookkeeping ───────────────────────────────────────────────────
        running_loss  += loss.item()
        total_batches += 1

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.5f}",
        )

    scheduler.step()

    avg_loss = running_loss / total_batches
    print(f"  → Epoch {epoch:02d} complete | Avg Loss: {avg_loss:.4f} "
          f"| LR: {scheduler.get_last_lr()[0]:.6f}\n")

    # ── Checkpoint every 10 epochs ────────────────────────────────────────
    if epoch % 10 == 0:
        ckpt_path = os.path.join(SAVE_DIR, f"resnet50_at_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "avg_loss"   : avg_loss,
            },
            ckpt_path,
        )
        print(f"  [CHECKPOINT] Saved → {ckpt_path}\n")

print("=" * 65)
print("  Adversarial Training Complete.")
print("=" * 65)
