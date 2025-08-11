# train_action_recognizer.py (V3.0 - Config-Driven with Early Stopping)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
import sys
import yaml
import argparse
from sklearn.model_selection import train_test_split

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subway_surfers_ai.action_recognition.model import ActionRecognitionCNN
from subway_surfers_ai.action_recognition.dataloader import ActionRecognitionDataset

def train(config_path):
    """Main training and validation loop, driven by a YAML config file."""
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_cfg = config.get('data_params', {})
    train_cfg = config.get('train_params', {})
    model_cfg = config.get('model_params', {})

    device = train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    MODEL_SAVE_DIR = PROJECT_ROOT / "models"
    BEST_MODEL_PATH = MODEL_SAVE_DIR / "action_recognizer_best.pt"
    MODEL_SAVE_DIR.mkdir(exist_ok=True)

    print("--- Starting Action Recognition Seed Model Training (Config-Driven) ---")
    print(f"Using device: {device}")

    # 2. Initialize and Split Dataset
    ANNOTATION_PATH = PROJECT_ROOT / "data" / "dongzuo_shibie_shuju" / "annotations" / "dongzuo_biaozhu.json"
    RAW_FRAMES_DIR = PROJECT_ROOT / "data" / "dongzuo_shibie_shuju" / "raw_frames"
    
    full_dataset = ActionRecognitionDataset(
        ANNOTATION_PATH, 
        RAW_FRAMES_DIR,
        zhen_geshu=model_cfg.get('num_frames', 8),
        tupian_chicun=(model_cfg.get('image_size', 64), model_cfg.get('image_size', 64))
    )
        
    indices = list(range(len(full_dataset)))
    labels = [item['action'] for item in full_dataset.biaozhu_liebiao]
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg.get('batch_size', 16), shuffle=True, num_workers=data_cfg.get('num_workers', 4))
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.get('batch_size', 16), shuffle=False, num_workers=data_cfg.get('num_workers', 4))
    print(f"Datasets loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # 3. Initialize Model, Loss, and Optimizer
    model = ActionRecognitionCNN(
        num_frames=model_cfg.get('num_frames', 8),
        num_classes=model_cfg.get('num_classes', 4)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.get('learning_rate', 0.001))
    
    # 4. Early Stopping Setup
    patience = train_cfg.get('early_stopping_patience', 10)
    metric = train_cfg.get('early_stopping_metric', 'val_accuracy')
    mode = train_cfg.get('early_stopping_mode', 'max')
    
    best_metric_value = -float('inf') if mode == 'max' else float('inf')
    patience_counter = 0
    best_model_state = None

    # 5. Training & Validation Loop
    for epoch in range(train_cfg.get('epochs', 50)):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.get('epochs', 50)} [Training]")
        for data, labels in train_pbar:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation Phase ---
        model.eval()
        total_val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1:03d} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # --- Early Stopping Logic ---
        current_metric_value = val_accuracy if metric == 'val_accuracy' else avg_val_loss
        
        if (mode == 'max' and current_metric_value > best_metric_value) or \
           (mode == 'min' and current_metric_value < best_metric_value):
            best_metric_value = current_metric_value
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f"  \033[92m(New best metric: {best_metric_value:.4f}! Saving model...)\033[0m")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  \033[93m(Early stopping triggered after {patience} epochs with no improvement.)\033[0m")
                break
    
    # 6. Save the best model
    if best_model_state:
        torch.save(best_model_state, BEST_MODEL_PATH)
        print(f"\n--- Training Finished! Best model saved to {BEST_MODEL_PATH} ---")
        print(f"Best validation metric ({metric}): {best_metric_value:.4f}")
    else:
        print("\n--- Training Finished! No best model was saved (no improvement observed). ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the action recognition model.")
    parser.add_argument("--config", type=str, default="action_recognizer_config.yaml",
                        help="Path to the training configuration file.")
    args = parser.parse_args()
    train(args.config)