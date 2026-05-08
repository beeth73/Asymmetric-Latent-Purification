# =============================================================================
# CELL 4 — RESTART: Per-epoch concatenated checkpointing
# =============================================================================
#
# CHANGES FROM PREVIOUS VERSION:
#   • Saves a checkpoint after EVERY epoch (not every 10)
#   • All epochs concatenated into ONE file: resnet50_at_all_epochs.pth
#     Structure: { "epochs": [ {epoch, model_state, optim_state, ...}, ... ] }
#   • On restart, detects existing file and RESUMES from last saved epoch
#     so you never lose progress if the kernel dies again
#   • Emergency save: catches KeyboardInterrupt (Ctrl+C / kernel interrupt)
#     and flushes current state before exiting cleanly
#   • Separate latest.pth always mirrors the most recent epoch (fast reload)
# =============================================================================

import os, gc, time, signal
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm

# ── Stability flags (same as before) ─────────────────────────────────────────
torch.backends.cudnn.enabled       = False
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = False
torch.use_deterministic_algorithms(False)
torch.cuda.empty_cache()
gc.collect()

DEVICE   = torch.device("cuda")
SAVE_DIR = "./Paper2_V100_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

CONCAT_PATH  = os.path.join(SAVE_DIR, "resnet50_at_all_epochs.pth")
LATEST_PATH  = os.path.join(SAVE_DIR, "resnet50_at_latest.pth")
NUM_EPOCHS   = 50
EPS          = 8  / 255.0
ALPHA        = 3  / 255.0
STEPS        = 3

# ── Model ─────────────────────────────────────────────────────────────────────
classifier    = models.resnet50(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 10)
classifier    = classifier.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(),
                      lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# ── Resume detection ──────────────────────────────────────────────────────────
start_epoch  = 1
history      = []   # list of per-epoch dicts — the concat log

if os.path.exists(CONCAT_PATH):
    print(f"  [RESUME] Found existing checkpoint file: {CONCAT_PATH}")
    existing = torch.load(CONCAT_PATH, map_location=DEVICE)
    history  = existing.get("epochs", [])

    if history:
        last = history[-1]
        classifier.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optim_state"])
        scheduler.load_state_dict(last["sched_state"])
        start_epoch = last["epoch"] + 1
        print(f"  [RESUME] Resuming from epoch {last['epoch']} "
              f"(avg_loss={last['avg_loss']:.4f})")
    else:
        print(f"  [RESUME] File exists but no epochs found — starting fresh.")
else:
    print(f"  [FRESH ] No checkpoint found — starting from scratch.")

if start_epoch > NUM_EPOCHS:
    print(f"  [DONE  ] All {NUM_EPOCHS} epochs already completed. Nothing to do.")
    raise SystemExit(0)

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_epoch(epoch, avg_loss):
    """Append this epoch's state to the concat file and update latest."""
    entry = {
        "epoch"      : epoch,
        "model_state": {k: v.cpu() for k, v in classifier.state_dict().items()},
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "avg_loss"   : avg_loss,
        "timestamp"  : time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    history.append(entry)

    # Atomic write — write to .tmp then rename so a crash never corrupts
    tmp = CONCAT_PATH + ".tmp"
    torch.save({"epochs": history}, tmp)
    os.replace(tmp, CONCAT_PATH)

    # latest.pth = just the most recent entry (fast single-epoch reload)
    tmp2 = LATEST_PATH + ".tmp"
    torch.save(entry, tmp2)
    os.replace(tmp2, LATEST_PATH)

    print(f"  [SAVED ] Epoch {epoch:02d} → {CONCAT_PATH}  "
          f"({len(history)} epochs stored, "
          f"file size: {os.path.getsize(CONCAT_PATH)/1e6:.1f} MB)")

# ── PGD-3 attack (unchanged) ──────────────────────────────────────────────────
def pgd_attack(model, x, y, eps=EPS, alpha=ALPHA, steps=STEPS):
    x = x.detach().float()
    y = y.detach()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x + delta, 0.0, 1.0)
    model.eval()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = criterion(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv,
                                   retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta,  min=0.0,  max=1.0)
    model.train()
    return x_adv.detach()

# ── Training loop ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  PGD-3 AT | ResNet-50 | CIFAR-10 | Epochs {start_epoch}–{NUM_EPOCHS}")
print(f"  Checkpointing: every epoch → {CONCAT_PATH}")
print("=" * 65 + "\n")

interrupted = False

try:
    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        classifier.train()
        running_loss  = 0.0
        total_batches = 0
        t0            = time.time()

        loop = tqdm(train_loader,
                    desc=f"Epoch [{epoch:02d}/{NUM_EPOCHS}]",
                    leave=True, dynamic_ncols=True)

        for images, labels in loop:
            images = images.to(DEVICE, dtype=torch.float32, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            adv_images = pgd_attack(classifier, images, labels)

            classifier.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = classifier(adv_images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss  += loss.item()
            total_batches += 1
            loop.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.5f}")

        scheduler.step()

        avg_loss   = running_loss / total_batches
        epoch_time = time.time() - t0
        print(f"  → Epoch {epoch:02d} | Loss: {avg_loss:.4f} "
              f"| LR: {scheduler.get_last_lr()[0]:.6f} "
              f"| {epoch_time:.1f}s")

        # Save after EVERY epoch — appended to concat file
        save_epoch(epoch, avg_loss)

except KeyboardInterrupt:
    # Kernel interrupt (Stop button) — flush state before dying
    interrupted = True
    print(f"\n  [INTERRUPT] Caught stop signal — flushing emergency save...")
    if total_batches > 0:
        avg_loss = running_loss / total_batches
        save_epoch(epoch, avg_loss)
        print(f"  [INTERRUPT] Emergency save complete at epoch {epoch}.")

finally:
    if not interrupted:
        print("\n" + "=" * 65)
        print(f"  ✅  Training complete. {len(history)} epochs in {CONCAT_PATH}")
        print("=" * 65)

# ── How to reload any specific epoch later ────────────────────────────────────
# data   = torch.load("./Paper2_V100_Results/resnet50_at_all_epochs.pth")
# ep30   = next(e for e in data["epochs"] if e["epoch"] == 30)
# classifier.load_state_dict(ep30["model_state"])
