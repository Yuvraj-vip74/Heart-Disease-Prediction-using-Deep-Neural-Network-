# train.py

# ── DEVICE SETUP ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")
if device.type == 'cuda':
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")

# ── LOAD PREPROCESSED DATA ───────────────────────────────────
print("\nLoading preprocessed data...")
X_train = np.load('X_train.npy').astype(np.float32)
X_test  = np.load('X_test.npy').astype(np.float32)
y_train = np.load('y_train.npy').astype(np.int64)
y_test  = np.load('y_test.npy').astype(np.int64)
print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")

INPUT_SIZE = X_train.shape[1]

# ── PYTORCH DATASETS & DATALOADERS ───────────────────────────
train_dataset = TensorDataset(
    torch.tensor(X_train), torch.tensor(y_train))
test_dataset  = TensorDataset(
    torch.tensor(X_test),  torch.tensor(y_test))

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True,  drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                           shuffle=False)

# ── MODEL, LOSS, OPTIMISER ───────────────────────────────────
model     = HeartNet(input_size=INPUT_SIZE).to(device)
class_weights = torch.tensor([1.2, 0.8]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimiser = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', patience=3, factor=0.5
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel architecture loaded — {total_params:,} parameters")

# ── TRAINING LOOP ─────────────────────────────────────────────
EPOCHS = 100

train_losses, test_losses     = [], []
train_accuracies, test_accuracies = [], []

print(f"\nTraining for {EPOCHS} epochs on {device}...\n")
print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  "
      f"{'Test Loss':>10}  {'Test Acc':>9}")
print("-" * 56)

for epoch in range(1, EPOCHS + 1):

    # ── Train ──────────────────────────────────────────────
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimiser.zero_grad()
        out  = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * Xb.size(0)
        preds        = out.argmax(dim=1)
        correct      += (preds == yb).sum().item()
        total        += Xb.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total * 100
   

    # ── Evaluate ────────────────────────────────────────────
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out    = model(Xb)
            loss   = criterion(out, yb)
            val_loss    += loss.item() * Xb.size(0)
            preds        = out.argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total   += Xb.size(0)

    test_loss = val_loss / val_total
    test_acc  = val_correct / val_total * 100
    scheduler.step(test_loss)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:>6}  {train_loss:>11.4f}  {train_acc:>9.2f}%  "
              f"{test_loss:>10.4f}  {test_acc:>8.2f}%")

# ── FINAL EVALUATION ─────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        out = model(Xb)
        all_preds.extend(out.argmax(dim=1).cpu().numpy())
        all_labels.extend(yb.numpy())

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec  = recall_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds)
cm   = confusion_matrix(all_labels, all_preds)

print("\n" + "=" * 56)
print("FINAL TEST METRICS")
print("=" * 56)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1-Score  : {f1*100:.2f}%")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nOverfitting gap : "
      f"{train_accuracies[-1] - test_accuracies[-1]:.2f}%")

# ── PLOTS ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('HeartNet Training Results', fontsize=14, fontweight='bold')

# Loss curve
axes[0].plot(train_losses, 'r-o', label='Training Loss', markersize=3)
axes[0].plot(test_losses,  'b-o', label='Validation Loss', markersize=3)
axes[0].set_title('Loss Over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy curve
axes[1].plot(train_accuracies, 'r-o', label='Training Accuracy', markersize=3)
axes[1].plot(test_accuracies,  'b--o',
             label=f'Test Accuracy ({test_accuracies[-1]:.2f}%)', markersize=3)
axes[1].set_title('Accuracy Over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['No Disease (0)', 'Disease (1)'],
            yticklabels=['No Disease (0)', 'Disease (1)'])
axes[2].set_title('Confusion Matrix — Test Set')
axes[2].set_xlabel('Predicted Label')
axes[2].set_ylabel('Actual Label')

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved → training_results.png")

# ── SAVE MODEL ────────────────────────────────────────────────
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size'      : INPUT_SIZE,
    'test_accuracy'   : acc * 100,
    'epochs'          : EPOCHS,
}, 'heart_model.pth')
print("Model saved → heart_model.pth")
print("\n✅  Training complete!")
