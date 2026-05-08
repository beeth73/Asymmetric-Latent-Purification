from torch.cuda.amp import autocast, GradScaler

# 1. THE ATTACKER (Optimized for Training)
def pgd_attack_train(model, x, y, eps=8/255, alpha=2/255, iters=7):
    model.eval()
    x_adv = x.clone().detach() + torch.FloatTensor(x.shape).uniform_(-eps, eps).to(device)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    for _ in range(iters):
        x_adv.requires_grad = True
        with autocast(): # Faster gradient calculation
            output = model(x_adv)
            loss = nn.CrossEntropyLoss()(output, y)
        
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

# 2. THE TRAINING ENGINE
scaler = GradScaler() # For Mixed Precision
optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50) # 50 Epochs is enough for a V100
criterion = nn.CrossEntropyLoss()

print(f"🔥 [IGNITION] Starting 50 Epochs of AT on V100s. Deadline: 4:00 PM.")

def run_adversarial_training(epochs=50):
    start_time = time.time()
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        
        # tqdm for visual feedback
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Generate Attack
            data_adv = pgd_attack_train(classifier, data, target)
            
            optimizer.zero_grad()
            with autocast(): # SPEED BOOST: Mixed Precision
                output = classifier(data_adv)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        
        # 💾 THE HEARTBEAT SAVER: Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"resnet50_at_epoch_{epoch+1}.pth")
            torch.save(classifier.state_dict(), checkpoint_path)
            print(f"\n✅ [CHECKPOINT] {checkpoint_path} saved to RAID array.")

    # FINAL SAVE
    torch.save(classifier.state_dict(), os.path.join(SAVE_DIR, "resnet50_at_final.pth"))
    print(f"🏆 [MISSION SUCCESS] Final model saved. Total Time: {(time.time()-start_time)/60:.2f} mins")

# LAUNCH THE ENGINE
run_adversarial_training()
