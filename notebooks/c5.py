# =============================================================================
# CELL 5 — GATE 1: ROBUSTNESS VERIFICATION  (per-batch save + resume)
# =============================================================================
#
# CHANGES FROM PREVIOUS VERSION:
#   • Saves after EVERY batch — appended to one concat file
#   • Auto-resumes if interrupted — skips already-evaluated batches
#   • Atomic writes (.tmp → rename) — crash-safe at all times
#   • KeyboardInterrupt caught — flushes partial batch before exit
#   • Model restored to train() at end regardless of how exit happens
#   • All 5 original bugs fixed (dtype, model.train, sample count,
#     PGD steps, threshold scaling)
# =============================================================================

import os, time
import torch
import torch.nn.functional as F
from tqdm import tqdm

SAVE_DIR    = "./Paper2_V100_Results"
GATE1_PATH  = os.path.join(SAVE_DIR, "gate1_results.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Eval config ───────────────────────────────────────────────────────────────
CHECKPOINT_EPOCH = 50    # ← change if evaluating a mid-training checkpoint
EPS              = 8  / 255.0
ALPHA            = 2  / 255.0   # eps/4 — standard PGD eval step size
STEPS            = 50           # strong eval (literature standard)
N_BATCHES        = 8            # 8 × 128 = 1,024 images

# ── Threshold scaling (fair at any checkpoint epoch) ─────────────────────────
frac             = min(CHECKPOINT_EPOCH / 50.0, 1.0)
CLEAN_TARGET     = 60 + frac * 22    # 60% @ ep1 → 82% @ ep50
ROBUST_TARGET    = 15 + frac * 25    # 15% @ ep1 → 40% @ ep50

# ── Resume detection ──────────────────────────────────────────────────────────
batches_done    = []   # list of per-batch result dicts already evaluated
clean_correct   = 0
robust_correct  = 0
total_images    = 0

if os.path.exists(GATE1_PATH):
    existing = torch.load(GATE1_PATH, map_location="cpu")
    batches_done   = existing.get("batches", [])
    clean_correct  = existing.get("clean_correct",  0)
    robust_correct = existing.get("robust_correct", 0)
    total_images   = existing.get("total_images",   0)
    print(f"  [RESUME] Found gate1_results.pth — "
          f"{len(batches_done)} batches already done "
          f"({total_images} images evaluated).")
else:
    print(f"  [FRESH ] No prior gate1 results — starting from scratch.")

already_done = len(batches_done)

# ── Save helper ───────────────────────────────────────────────────────────────
def save_gate1():
    payload = {
        "batches"        : batches_done,
        "clean_correct"  : clean_correct,
        "robust_correct" : robust_correct,
        "total_images"   : total_images,
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "eps"            : EPS,
        "alpha"          : ALPHA,
        "steps"          : STEPS,
        "timestamp"      : time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = GATE1_PATH + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, GATE1_PATH)

# ── PGD attack (uses same function defined in Cell 4) ─────────────────────────
# pgd_attack() is already in scope from Cell 4 — no redefinition needed.

# ── Evaluation loop ───────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Gate 1 — Robustness Evaluation")
print(f"  Checkpoint epoch : {CHECKPOINT_EPOCH}/50")
print(f"  PGD              : eps={EPS:.4f}  alpha={ALPHA:.4f}  steps={STEPS}")
print(f"  Images target    : {N_BATCHES} batches × {test_loader.batch_size} = "
      f"{N_BATCHES * test_loader.batch_size}")
print(f"  Targets          : clean>{CLEAN_TARGET:.0f}%  robust>{ROBUST_TARGET:.0f}%")
if already_done:
    print(f"  Resuming from    : batch {already_done + 1}")
print(f"{'='*55}\n")

classifier.eval()
interrupted = False

try:
    pbar = tqdm(enumerate(test_loader),
                total=N_BATCHES,
                initial=already_done,
                desc="Gate 1 eval")

    for i, (images, labels) in pbar:
        if i >= N_BATCHES:
            break

        # Skip batches already evaluated before the interrupt
        if i < already_done:
            continue

        t_batch = time.time()

        # dtype fix — required with cudnn.enabled=False
        images = images.to(DEVICE, dtype=torch.float32, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        batch_size = images.size(0)

        # ── Clean accuracy ────────────────────────────────────────────────
        with torch.no_grad():
            outputs = classifier(images)
            batch_clean = (outputs.argmax(1) == labels).sum().item()

        # ── Robust accuracy (PGD-50) ──────────────────────────────────────
        adv_images = pgd_attack(classifier, images, labels,
                                eps=EPS, alpha=ALPHA, steps=STEPS)
        with torch.no_grad():
            adv_outputs = classifier(adv_images)
            batch_robust = (adv_outputs.argmax(1) == labels).sum().item()

        # ── Accumulate ────────────────────────────────────────────────────
        clean_correct  += batch_clean
        robust_correct += batch_robust
        total_images   += batch_size

        batch_entry = {
            "batch_idx"    : i,
            "batch_size"   : batch_size,
            "clean_correct": batch_clean,
            "robust_correct": batch_robust,
            "clean_acc"    : 100.0 * batch_clean  / batch_size,
            "robust_acc"   : 100.0 * batch_robust / batch_size,
            "elapsed_s"    : round(time.time() - t_batch, 2),
        }
        batches_done.append(batch_entry)

        # ── Save after every batch ────────────────────────────────────────
        save_gate1()

        # Running totals in progress bar
        run_clean  = 100.0 * clean_correct  / total_images
        run_robust = 100.0 * robust_correct / total_images
        pbar.set_postfix(
            clean=f"{run_clean:.1f}%",
            robust=f"{run_robust:.1f}%",
            saved=f"batch {i+1}/{N_BATCHES}",
        )

except KeyboardInterrupt:
    interrupted = True
    print(f"\n  [INTERRUPT] Stop signal — progress saved up to batch "
          f"{len(batches_done)}/{N_BATCHES}  ({total_images} images).")
    print(f"  Re-run this cell to resume from batch {len(batches_done) + 1}.")

finally:
    # Always restore train mode regardless of how we exit
    classifier.train()

# ── Final verdict (only if we finished all batches) ───────────────────────────
if not interrupted and total_images > 0:
    clean_acc  = 100.0 * clean_correct  / total_images
    robust_acc = 100.0 * robust_correct / total_images

    print(f"\n{'='*55}")
    print(f"  Final Results  ({total_images} images)")
    print(f"{'='*55}")
    print(f"  Clean  Accuracy : {clean_acc:.2f}%   (target >{CLEAN_TARGET:.0f}%)")
    print(f"  Robust Accuracy : {robust_acc:.2f}%   (target >{ROBUST_TARGET:.0f}%)")
    print(f"  Saved to        : {GATE1_PATH}")
    print(f"{'='*55}")

    clean_pass  = clean_acc  > CLEAN_TARGET
    robust_pass = robust_acc > ROBUST_TARGET

    if clean_pass and robust_pass:
        print(f"\n  ✅  GATE 1 PASSED — model is battle-hardened.")
    elif not clean_pass and not robust_pass:
        print(f"\n  ❌  GATE 1 FAILED — both metrics below target.")
        print(f"      Check: LR schedule, normalisation, batch size,")
        print(f"      and whether epoch {CHECKPOINT_EPOCH} has converged.")
    elif not clean_pass:
        print(f"\n  ⚠️   Clean accuracy low — possible catastrophic forgetting.")
        print(f"      Consider adding clean examples to the training mix.")
    else:
        print(f"\n  ⚠️   Robust accuracy low — PGD-3 attack may be too weak.")
        print(f"      Consider increasing training attack steps or epsilon.")

    # Persist final summary into the save file
    save_gate1()

elif not interrupted:
    print("  [WARN] No images evaluated — check test_loader.")

# ── How to reload results later ───────────────────────────────────────────────
# data = torch.load("./Paper2_V100_Results/gate1_results.pth")
# for b in data["batches"]:
#     print(b["batch_idx"], b["clean_acc"], b["robust_acc"])
