# Trainer Class
#   Save model file per training -> models/[model ver]/bak/*.pth,models/[model ver]/bak/*_info.json,models/[model ver]/best.pth,models/[model ver]/best_info.json (Load best.pth before learning (if you change model archtecture, you should copy and store model.py in models/[model ver]/))
#   Save Log per training and plot -> logs/YYYYMMDDHHmm
# Validater Class
#   Save Log per training and plot -> logs/YYYYMMDDHHmm
# Read Log File after validate -> if model accuracy is OK? (trainer, validater loss is lowest, if not true => you refine model archtecture, if true => you refine model size)
# **↑このコメントは残すこと**
from dataclasses import dataclass, asdict
from mixup import mixup_criterion, mixup_data
from typing import Optional
import os
import csv
import json
import shutil
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from dataset import get_dataloaders, CLASSES, NUM_CLASSES
from model import DrumResNet

@dataclass
class ModelInfo:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float]
    val_accuracy: Optional[float]
    timestamp: str
    log_dir: Optional[str]

class Trainer:
    def __init__(self, model, dataloader, criterion, optimizer, scheduler, model_ver="v1", log_dir="logs", device: str|torch.device="cpu"):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_ver = model_ver
        self.device = device
        self.best_info: Optional[ModelInfo] = None
        
        # Setup model directories
        self.model_dir = os.path.join("models", self.model_ver)
        self.bak_dir = os.path.join(self.model_dir, "bak")
        os.makedirs(self.bak_dir, exist_ok=True)

        # Load best_info.json before learning
        best_info_path = os.path.join(self.model_dir, "best_info.json")
        if os.path.exists(best_info_path):
            print(f"Loading {best_info_path} before learning...")
            with open(best_info_path, "r") as f:
                info_dict = json.load(f)
                self.best_info = ModelInfo(**info_dict)

        # Load best.pth before learning
        best_model_path = os.path.join(self.model_dir, "best.pth")
        if os.path.exists(best_model_path):
            print(f"Loading {best_model_path} before learning...")
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Loaded model, optimizer, and scheduler states.")
            else:
                self.model.load_state_dict(checkpoint)
                print("Loaded legacy model state (model weights only).")
                if self.best_info is not None:
                    start_epoch = self.best_info.epoch + 1
                    self.optimizer.zero_grad()
                    self.optimizer.step()
                    for _ in range(start_epoch - 1):
                        self.scheduler.step()
        
        # Setup log directory
        if log_dir is None:
            log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d%H%M"))
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "train_log.csv")
        
        # Copy past log if exists
        if self.best_info is not None and self.best_info.log_dir is not None:
            old_log_file = os.path.join(self.best_info.log_dir, "train_log.csv")
            if os.path.exists(old_log_file):
                shutil.copy(old_log_file, self.log_file)
        
        # Initialize log
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "loss", "accuracy"])

        # Copy and store model.py
        if os.path.exists("model.py"):
            shutil.copy("model.py", os.path.join(self.model_dir, "model.py"))

    def get_model_info(self) -> Optional[ModelInfo]:
        return self.best_info

    def save_model(self, model_info: ModelInfo, is_best=False):
        epoch = model_info.epoch
        pth_path = os.path.join(self.bak_dir, f"epoch_{epoch}.pth")
        json_path = os.path.join(self.bak_dir, f"epoch_{epoch}_info.json")
        
        info_dict = asdict(model_info)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        
        torch.save(checkpoint, pth_path)
        with open(json_path, "w") as f:
            json.dump(info_dict, f, indent=4)
            
        if is_best:
            best_pth_path = os.path.join(self.model_dir, "best.pth")
            best_json_path = os.path.join(self.model_dir, "best_info.json")
            torch.save(checkpoint, best_pth_path)
            with open(best_json_path, "w") as f:
                json.dump(info_dict, f, indent=4)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # --- Mixupの適用 ---
            # 50%の確率でMixupを適用（常に混ぜると純粋なクラスの特徴が薄れるため）
            if torch.rand(1).item() < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                
                loss.backward()
                self.optimizer.step()
                
                # 精度計算の近似 (ブレンド比率が高い方のラベルを正解とみなす)
                _, predicted = outputs.max(1)
                correct += (lam * predicted.eq(targets_a).float() + (1 - lam) * predicted.eq(targets_b).float()).sum().item()
                
            else:
                # 通常の学習
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            # -------------------
            
            """self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()"""
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            """correct += predicted.eq(targets).sum().item()"""
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Save Log
        with open(self.log_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_loss, epoch_acc])
        
        print(f"Train Epoch: {epoch} \tLoss: {epoch_loss:.6f} \tAcc: {epoch_acc:.4f}")
        return epoch_loss, epoch_acc
        
    def plot_logs(self):
        val_log_file = os.path.join(self.log_dir, "val_log.csv")
        epochs, train_losses, train_accs = [], [], []
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row["epoch"]))
                    train_losses.append(float(row["loss"]))
                    train_accs.append(float(row["accuracy"]))
                    
        val_epochs, val_losses, val_accs = [], [], []
        if os.path.exists(val_log_file):
            with open(val_log_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    val_epochs.append(int(row["epoch"]))
                    val_losses.append(float(row["loss"]))
                    val_accs.append(float(row["accuracy"]))

        if not epochs:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(epochs, train_losses, label="Train Loss")
        if val_epochs:
            ax1.plot(val_epochs, val_losses, label="Val Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        
        ax2.plot(epochs, train_accs, label="Train Acc")
        if val_epochs:
            ax2.plot(val_epochs, val_accs, label="Val Acc")
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        
        plt.savefig(os.path.join(self.log_dir, "training_metrics.png"))
        plt.close()


class Validater:
    def __init__(self, model, dataloader, criterion, log_dir=None, device: str | torch.device="cpu", model_info: Optional[ModelInfo] = None):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        
        if log_dir is None:
            log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d%H%M"))
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "val_log.csv")
        self.class_log_file = os.path.join(self.log_dir, "val_class_log.csv")
        self.cm_log_file = os.path.join(self.log_dir, "val_confusion_matrix.csv")
        
        # Copy past log if exists
        if model_info is not None and model_info.log_dir is not None:
            old_log_file = os.path.join(model_info.log_dir, "val_log.csv")
            if os.path.exists(old_log_file):
                shutil.copy(old_log_file, self.log_file)
            
            old_class_log_file = os.path.join(model_info.log_dir, "val_class_log.csv")
            if os.path.exists(old_class_log_file):
                shutil.copy(old_class_log_file, self.class_log_file)
                
            old_cm_log_file = os.path.join(model_info.log_dir, "val_confusion_matrix.csv")
            if os.path.exists(old_cm_log_file):
                shutil.copy(old_cm_log_file, self.cm_log_file)
        
        # Initialize log
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "loss", "accuracy"])

        # Initialize class log
        if not os.path.exists(self.class_log_file):
            with open(self.class_log_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch"] + CLASSES)

    def plot_and_log_confusion_matrix(self, all_targets, all_predictions, epoch):
        cm = confusion_matrix(all_targets, all_predictions, labels=range(NUM_CLASSES))
        
        # 行（真のクラス）ごとに正規化し、推定割合を算出（ゼロ除算回避のため微小値を加算）
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

        # マトリクスプロットの描画
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1) # type: ignore
        fig.colorbar(cax)

        ax.set_xticks(np.arange(NUM_CLASSES))
        ax.set_yticks(np.arange(NUM_CLASSES))
        ax.set_xticklabels(CLASSES, rotation=45, ha="left")
        ax.set_yticklabels(CLASSES)
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_title(f'Normalized Confusion Matrix (Epoch {epoch})')

        # マス目に数値をテキストとして追加
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "confusion_matrix.png"))
        plt.close()

        # CSVへログとして出力
        file_exists = os.path.exists(self.cm_log_file)
        with open(self.cm_log_file, "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epoch", "true_class"] + [f"pred_{c}" for c in CLASSES])
            for i in range(NUM_CLASSES):
                row = [epoch, CLASSES[i]] + [f"{cm_norm[i, j]:.4f}" for j in range(NUM_CLASSES)]
                writer.writerow(row)

    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        class_correct = [0] * NUM_CLASSES
        class_total = [0] * NUM_CLASSES
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 混同行列のためにターゲットと予測結果を記録
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                # クラスごとの集計
                c = predicted.eq(targets)
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # クラスごとの精度計算
        class_accs = []
        for i in range(NUM_CLASSES):
            acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            class_accs.append(acc)
        
        # Save Log
        with open(self.log_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_loss, epoch_acc])
            
        with open(self.class_log_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch] + class_accs)
            
        # 混同行列の描画とログ出力
        self.plot_and_log_confusion_matrix(all_targets, all_predictions, epoch)
            
        print(f"Val Epoch: {epoch} \tLoss: {epoch_loss:.6f} \tAcc: {epoch_acc:.4f}")
        return epoch_loss, epoch_acc

def analyze_logs(log_dir):
    """
    Read Log File after validate -> if model accuracy is OK?
    """
    val_log_file = os.path.join(log_dir, "val_log.csv")
    train_log_file = os.path.join(log_dir, "train_log.csv")
    
    if not os.path.exists(val_log_file) or not os.path.exists(train_log_file):
        print("Missing log files for analysis.")
        return
        
    val_losses = []
    with open(val_log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_losses.append(float(row["loss"]))
            
    train_losses = []
    with open(train_log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_losses.append(float(row["loss"]))
            
    if not val_losses or not train_losses:
        return
        
    current_val_loss = val_losses[-1]
    lowest_val_loss = min(val_losses)
    
    current_train_loss = train_losses[-1]
    lowest_train_loss = min(train_losses)
    
    print("\n--- Model Analysis ---")
    if current_val_loss <= lowest_val_loss and current_train_loss <= lowest_train_loss:
        print("Status: Model accuracy is OK. Trainer, validater loss is lowest.")
        print("Action: if true => you refine model size")
    else:
        print(f"Status: Validation or train loss has increased.")
        print("Action: if not true => you refine model archtecture")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Drum Detection Model")
    parser.add_argument("--model-ver", type=str, default="dualnorm_allnorm_v14", help="Model version name (e.g., dualnorm_allnorm_v14)")
    parser.add_argument("-d", "--data-dir", type=str, default="", help="Directory containing training audio files")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    args = parser.parse_args()

    model_ver = args.model_ver
    num_epochs = args.epochs
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print("Initializing components...")
    train_loader, val_loader = get_dataloaders(data_dir=args.data_dir,batch_size=32, val_split=0.2)
    
    model = DrumResNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d%H%M"))
    
    trainer = Trainer(model, train_loader, criterion, optimizer, scheduler=scheduler, model_ver=model_ver, log_dir=log_dir, device=device)
    validater = Validater(model, val_loader, criterion, log_dir=log_dir, device=device, model_info=trainer.get_model_info())
    
    # 学習の再開設定
    start_epoch = 1
    best_val_loss = float('inf')
    
    if trainer.best_info is not None:
        start_epoch = trainer.best_info.epoch + 1
        if trainer.best_info.val_loss is not None:
            best_val_loss = trainer.best_info.val_loss
        print(f"Resuming training from epoch {start_epoch}...")
    
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_acc = trainer.train_epoch(epoch)
        val_loss, val_acc = validater.validate_epoch(epoch)
        
        # Step the scheduler
        scheduler.step()
        
        # ModelInfoによる状態の管理と保存
        current_info = ModelInfo(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            timestamp=datetime.datetime.now().isoformat(),
            log_dir=log_dir
        )
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            
        trainer.save_model(current_info, is_best=is_best)
            
        trainer.plot_logs()
        analyze_logs(log_dir)