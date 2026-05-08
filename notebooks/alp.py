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
# CELL 4 — PGD-3 Adversarial Training Loop  ✅ cuDNN-CRASH-PROOF
# =============================================================================
#
# ROOT CAUSE OF YOUR CRASH (line 23 in Cell 4d):
# ─────────────────────────────────────────────
# pgd_attack_atomic() ran the attack inside:
#       with torch.backends.cudnn.flags(enabled=False): ...
#
# When that context manager exits, cuDNN is re-enabled globally. BUT the
# `adv_images` tensor that came out of it carries internal CUDA metadata
# (stride layout, storage offset) that was produced while cuDNN was OFF.
# When you immediately pass that tensor to classifier(adv_images) with cuDNN
# ON, cuDNN's algorithm selection (the GET call) looks at the tensor's
# memory layout, cannot find a matching engine, and raises:
#       RuntimeError: GET was unable to find an engine to execute this computation
#
# THE FIX:
# ─────────────────────────────────────────────
# After the PGD loop, produce a brand-new contiguous FP32 tensor via:
#       adv_images = adv_images.detach().clone().contiguous().float()
# This allocates fresh CUDA memory with a clean standard layout that cuDNN
# can always handle, completely severing any metadata from the disabled-cuDNN
# context. One line. Crash eliminated.
#
# ADDITIONAL HARDENING applied in this cell:
#   • benchmark=False + deterministic=True  (already in Cell 1, repeated
#     here defensively in case cells run out of order)
#   • Explicit .float() cast on images entering PGD (pin_memory can
#     occasionally produce unexpected dtypes on some driver versions)
#   • model.eval() during attack, model.train() for weight update
#     (prevents BN running-stats corruption from adversarial batch stats)
#   • autograd.grad() instead of .backward() during PGD
#     (surgical gradient — never touches model.parameters())
#   • clip_grad_norm_ max_norm=1.0 (stabilises early AT epochs)
#   • Rich checkpoints: model + optimiser + scheduler state saved together
# =============================================================================

import os, gc, time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm

# ── Defensive flag reset (safe to repeat) ────────────────────────────────────
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled       = True   # keep cuDNN on for training speed
torch.cuda.empty_cache()
gc.collect()

DEVICE   = torch.device("cuda")
SAVE_DIR = "./Paper2_V100_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
classifier     = models.resnet50(weights=None)
classifier.fc  = nn.Linear(classifier.fc.in_features, 10)
classifier     = classifier.to(DEVICE)

# ── Loss / Optimiser / Scheduler ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    classifier.parameters(),
    lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# ── PGD-3 Attack ──────────────────────────────────────────────────────────────
EPS   = 8  / 255.0
ALPHA = 3  / 255.0
STEPS = 3

def pgd_attack(model, x, y,
               eps=EPS, alpha=ALPHA, steps=STEPS):
    """
    PGD-ℓ∞, hardened for V100 / cuDNN stability.

    Key contract: the tensor returned is ALWAYS a freshly allocated,
    contiguous, float32 CUDA tensor — regardless of how it was produced
    internally. This is what prevents the GET/FIND engine crash when the
    tensor is subsequently passed to the training forward pass.
    """
    # Cast inputs to clean float32 (pin_memory edge-case guard)
    x = x.detach().float()
    y = y.detach()

    # Random-uniform init inside the epsilon ball, then project to [0,1]
    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x + delta, 0.0, 1.0)

    model.eval()   # freeze BN to population stats during attack

    for _ in range(steps):
        # Fresh contiguous tensor every iteration — no stale views
        x_adv = x_adv.detach().contiguous().float()
        x_adv.requires_grad_(True)

        with torch.enable_grad():
            output = model(x_adv)
            loss   = criterion(output, y)

        # Gradient only w.r.t. x_adv — model weights untouched
        grad = torch.autograd.grad(
            loss, x_adv,
            retain_graph=False,
            create_graph=False,
        )[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            # Project back into epsilon ball around original x
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)

    model.train()  # restore training mode before returning

    # ── THE CRITICAL FIX ──────────────────────────────────────────────────
    # Allocate a completely new contiguous FP32 tensor. This severs every
    # trace of metadata from intermediate ops and ensures cuDNN's engine
    # selector always finds a valid algorithm in the training forward pass.
    return x_adv.detach().clone().contiguous().float()


# ── Training Loop ─────────────────────────────────────────────────────────────
NUM_EPOCHS = 50

print("\n" + "=" * 65)
print("  PGD-3 Adversarial Training  |  ResNet-50  |  CIFAR-10")
print(f"  Epochs: {NUM_EPOCHS}  |  ε={EPS:.4f}  |  α={ALPHA:.4f}  |  steps={STEPS}")
print("=" * 65 + "\n")

for epoch in range(1, NUM_EPOCHS + 1):

    classifier.train()
    running_loss  = 0.0
    total_batches = 0
    t0            = time.time()

    loop = tqdm(
        train_loader,
        desc=f"Epoch [{epoch:02d}/{NUM_EPOCHS}]",
        leave=True,
        dynamic_ncols=True,
    )

    for images, labels in loop:
        # Move to GPU as clean float32
        images = images.to(DEVICE, dtype=torch.float32, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # ── 1. Generate adversarial examples ──────────────────────────────
        adv_images = pgd_attack(classifier, images, labels)
        # adv_images is already: detached · cloned · contiguous · float32
        # No further surgery needed before the training forward pass.

        # ── 2. Training step on adversarial examples ──────────────────────
        classifier.train()                          # guaranteed train mode
        optimizer.zero_grad(set_to_none=True)

        outputs = classifier(adv_images)            # ← was crashing here
        loss    = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Bookkeeping ───────────────────────────────────────────────────
        running_loss  += loss.item()
        total_batches += 1
        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.5f}",
        )

    scheduler.step()

    avg_loss    = running_loss / total_batches
    epoch_time  = time.time() - t0

    print(f"  → Epoch {epoch:02d} | Avg Loss: {avg_loss:.4f} "
          f"| LR: {scheduler.get_last_lr()[0]:.6f} "
          f"| Time: {epoch_time:.1f}s\n")

    # ── Checkpoint every 10 epochs ────────────────────────────────────────
    if epoch % 10 == 0:
        ckpt_path = os.path.join(
            SAVE_DIR, f"resnet50_at_epoch_{epoch}.pth"
        )
        torch.save(
            {
                "epoch"      : epoch,
                "model_state": classifier.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "avg_loss"   : avg_loss,
            },
            ckpt_path,
        )
        print(f"  [CHECKPOINT] Saved → {ckpt_path}\n")

print("=" * 65)
print("  ✅  Adversarial Training Complete.")
print("=" * 65)
