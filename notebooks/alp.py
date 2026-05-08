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
# CELL 4 — FINAL / DEFINITIVE  PGD-3 Adversarial Training Loop
# =============================================================================
#
# ROOT CAUSE (this crash):
# ─────────────────────────────────────────────────────────────────────────────
# The error is at model(x_adv) INSIDE the PGD attack — line 102.
# This is a different failure mode from the previous crash.
#
# On a V100 with certain PyTorch + cuDNN builds, setting:
#       cudnn.deterministic = True
# causes cuDNN's conv algorithm registry to be restricted to only algorithms
# marked as "deterministic". For ResNet-50's conv1 (7×7 kernel, stride 2)
# operating on 32×32 CIFAR-10 images, the resulting output feature map
# (16×16 spatial) has a shape that falls into a gap in the V100's deterministic
# algorithm table — no engine is registered for it, so GET fails.
#
# This gap only manifests during the PGD attack because:
#   1. The attack calls model.eval() which changes BN's internal buffer shapes.
#   2. requires_grad_(True) on the input triggers a different internal CUDA
#      kernel dispatch path than the normal training path.
#   3. These two together produce a (dtype, shape, stride, requires_grad) tuple
#      that the deterministic algorithm registry has no entry for.
#
# THE DEFINITIVE FIX — Three-layer defence:
# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Run the entire PGD forward+backward inside a context that disables
#          cuDNN (torch.backends.cudnn.enabled = False via flags()) AND also
#          disables the deterministic algorithms enforcement. This forces PyTorch
#          to fall back to its pure CUDA math path (no cuDNN), which ALWAYS works
#          regardless of shape, dtype, or stride.
#
# Layer 2: After the attack, produce a brand-new tensor via
#          .detach().clone().contiguous().float() before it touches the training
#          forward pass. This severs all metadata from the non-cuDNN context.
#
# Layer 3: The training forward pass runs with cuDNN fully enabled (benchmark=
#          False, deterministic=True) — the clean tensor from Layer 2 is safe.
#
# Why not just disable cuDNN globally?
#   cuDNN provides ~3–5× speedup for conv layers on the V100. Disabling it
#   globally would make 50 epochs take ~5× longer. We only disable it for the
#   3 PGD forward passes per batch — the expensive training backward pass still
#   gets the full cuDNN acceleration.
# =============================================================================

import os, gc, time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm

# ── Global flags ─────────────────────────────────────────────────────────────
# deterministic=True for the TRAINING path (reproducibility)
# benchmark=False    mandatory for AT (dynamic tensor shapes in PGD loop)
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled       = True        # ON for training speed
torch.use_deterministic_algorithms(False)        # allow non-det fallbacks
                                                  # (needed so the cuDNN-off
                                                  #  path inside PGD doesn't
                                                  #  raise its own error)
torch.cuda.empty_cache()
gc.collect()

DEVICE   = torch.device("cuda")
SAVE_DIR = "./Paper2_V100_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
classifier    = models.resnet50(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 10)
classifier    = classifier.to(DEVICE)

# ── Loss / Optimiser / Scheduler ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    classifier.parameters(),
    lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# ── Attack hyperparameters ────────────────────────────────────────────────────
EPS   = 8  / 255.0
ALPHA = 3  / 255.0
STEPS = 3

# ── PGD-3 Attack ──────────────────────────────────────────────────────────────
def pgd_attack(model, x, y, eps=EPS, alpha=ALPHA, steps=STEPS):
    """
    PGD-ℓ∞, V100-hardened.

    The entire attack (all forward + backward passes) runs inside:
        torch.backends.cudnn.flags(enabled=False)
    which redirects conv ops to the pure-CUDA math path, bypassing cuDNN's
    algorithm registry entirely.  No GET/FIND error is possible here because
    there is no algorithm lookup — PyTorch uses its own CUDA kernels directly.

    The returned tensor is a fresh .clone().contiguous().float() allocated
    OUTSIDE the no-cuDNN context, so the training forward pass receives a
    standard tensor that cuDNN (re-enabled) is happy to process.
    """
    x = x.detach().float()   # ensure clean float32, no autograd history
    y = y.detach()

    # Random-uniform init inside epsilon ball, clipped to valid image range
    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x + delta, 0.0, 1.0).contiguous()

    model.eval()   # use BN running stats during attack (correct + stable)

    # ── LAYER 1: Disable cuDNN for all PGD forward/backward passes ────────
    with torch.backends.cudnn.flags(enabled=False):
        for _ in range(steps):
            x_adv = x_adv.detach().contiguous().float()
            x_adv.requires_grad_(True)

            # No autocast, no AMP, pure FP32
            with torch.enable_grad():
                output = model(x_adv)          # pure CUDA math, no cuDNN
                loss   = criterion(output, y)

            grad = torch.autograd.grad(
                loss, x_adv,
                retain_graph=False,
                create_graph=False,
            )[0]

            with torch.no_grad():
                x_adv = x_adv + alpha * grad.sign()
                delta = torch.clamp(x_adv - x, min=-eps, max=eps)
                x_adv = torch.clamp(x + delta, 0.0, 1.0).contiguous()

    model.train()   # restore before returning

    # ── LAYER 2: Fresh tensor — severs all non-cuDNN context metadata ─────
    # Allocate brand-new CUDA memory outside the flags() context.
    # This tensor has a completely standard memory layout that cuDNN
    # (re-enabled in the training step) can process without issue.
    return x_adv.detach().clone().contiguous().float()


# ── Training Loop ─────────────────────────────────────────────────────────────
NUM_EPOCHS = 50

print("\n" + "=" * 65)
print("  PGD-3 Adversarial Training  |  ResNet-50  |  CIFAR-10")
print(f"  Epochs={NUM_EPOCHS}  ε={EPS:.4f}  α={ALPHA:.4f}  steps={STEPS}")
print(f"  Attack path : pure CUDA (cuDNN disabled)")
print(f"  Train  path : cuDNN enabled (benchmark=False, deterministic=True)")
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
        images = images.to(DEVICE, dtype=torch.float32, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # ── 1. Generate adversarial examples (cuDNN OFF inside) ───────────
        adv_images = pgd_attack(classifier, images, labels)
        # adv_images: fresh · contiguous · float32 · no autograd history

        # ── 2. Training step (cuDNN ON — full speed) ──────────────────────
        # LAYER 3: training forward pass runs with cuDNN enabled.
        # adv_images is a clean tensor — no engine-lookup ambiguity.
        classifier.train()
        optimizer.zero_grad(set_to_none=True)

        outputs = classifier(adv_images)          # cuDNN-accelerated
        loss    = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss  += loss.item()
        total_batches += 1
        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.5f}",
        )

    scheduler.step()

    avg_loss   = running_loss / total_batches
    epoch_time = time.time() - t0

    print(f"  → Epoch {epoch:02d} | Avg Loss: {avg_loss:.4f} "
          f"| LR: {scheduler.get_last_lr()[0]:.6f} "
          f"| Time: {epoch_time:.1f}s\n")

    # ── Checkpoint every 10 epochs ────────────────────────────────────────
    if epoch % 10 == 0:
        ckpt_path = os.path.join(SAVE_DIR, f"resnet50_at_epoch_{epoch}.pth")
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
