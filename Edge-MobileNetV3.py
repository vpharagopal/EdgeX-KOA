import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, precision_score, 
                             recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, 
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
import cv2
from PIL import Image
cv2.setNumThreads(0)

print("Checking CUDA...", torch.cuda.is_available())
# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 10
LR = 5.0550803827429446e-05
WEIGHT_DECAY = 2.3708996227603637e-05
DROPOUT_RATE = 0.63
NUM_CLASSES = 5
CLASS_NAMES = ['KL-0', 'KL-1', 'KL-2', 'KL-3', 'KL-4']

# === CHANGE DATA_DIR TO THE ORIGINAL UNAUGMENTED FOLDER ===
DATA_DIR = r"C:\Users\vphar\KOA-Chen-Kaggle\archive" 
#RAD_WEIGHTS = r"C:\Users\vphar\KOA-Chen-Kaggle\archive\RadImageNet_pytorch\RadImageNet_pytorch\MobileNetV3.pt"
SAVE_DIR = r"C:\Users\vphar\KOA-Chen-Kaggle\PhD Objectives\Re-Execution-Without-BalancedSet-New\ImageNet\MobileNetV3-Img-Files"
PLOT_DIR = SAVE_DIR
# === EXPLICIT PATHS FOR RAW, PRE-SPLIT DATASET ===
TRAIN_DIR = r"C:\Users\vphar\KOA-Chen-Kaggle\archive\train"
VAL_DIR = r"C:\Users\vphar\KOA-Chen-Kaggle\archive\val"
TEST_DIR = r"C:\Users\vphar\KOA-Chen-Kaggle\archive\test"

# ==========================================
# 2. HELPER CLASSES
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ApplyCLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img.convert('L')) 
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_clahe = clahe.apply(img_np)
        return Image.fromarray(img_clahe).convert('RGB')


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.epochs_no_improve += 1
        
        if self.epochs_no_improve >= self.patience:
            return True
        return False

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        outputs = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(outputs, dim=1).item()
        
        self.model.zero_grad()
        outputs[0, class_idx].backward(retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        return heatmap, class_idx, outputs

# ==========================================
# 3. TRAINING & VALIDATION FUNCTIONS
# ==========================================
def get_model():
    # 1. Load the base MobileNetV3-Large model
    model = models.mobilenet_v3_large(weights='DEFAULT')
    
    # 2. Modify PyTorch's native classifier head for 5 classes
    # The default V3 classifier is: Linear(960->1280) -> Hardswish -> Dropout -> Linear(1280->1000)
    # We keep the optimized structure (and Hardswish), but update the dropout and final layer.
    model.classifier[2] = nn.Dropout(p=DROPOUT_RATE, inplace=True)
    model.classifier[3] = nn.Linear(in_features=1280, out_features=NUM_CLASSES)
    
    return model.to(device)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (inputs, labels) in enumerate(pbar, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f"{running_loss/batch_idx:.4f}", 'acc': f"{correct/total:.4f}"})
        
    return running_loss / len(loader), correct / total

def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets, all_preds = [], []
    
    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(pbar, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{running_loss/batch_idx:.4f}", 'acc': f"{correct/total:.4f}"})
            
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    epoch_kappa = cohen_kappa_score(all_targets, all_preds)
    epoch_mcc = matthews_corrcoef(all_targets, all_preds)
            
    return epoch_loss, epoch_acc, epoch_kappa, epoch_mcc

# ==========================================
# 4. MAIN EXECUTION PIPELINE (Single-Run)
# ==========================================
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("-> 1/3: Setting up Image Transforms (CLAHE)...")
    
    # --- 1. TRANSFORMS ---
    train_transform = transforms.Compose([
        ApplyCLAHE(clip_limit=2.0),          
        transforms.Resize((224, 224)), 
        transforms.RandomRotation(15),       
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((224, 224), scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_test_transform = transforms.Compose([
        ApplyCLAHE(clip_limit=2.0),          
        transforms.Resize((224, 224)),
        transforms.ToTensor(),               
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("-> 2/3: Scanning Windows folders for the Images... (This might take a minute)")
    # --- 2. DATA LOADERS (Using Original Splits) ---
    # No CV here: Direct training on Train folder, validation on Val folder
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)
    
    # num_workers=0 is the safest for Windows/Jupyter to prevent DataLoader hangs
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Datasets Loaded Successfully")
    print(f"Training: {len(train_dataset)} | Validation: {len(val_dataset)} | Independent Test: {len(test_dataset)}")

    print("-> 3/3: Downloading MobileNetV3 Weights from PyTorch...")
    # --- 3. MODEL INITIALIZATION ---
    model = get_model()
    optimizer = optim.RMSprop(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(gamma=2.0)
    early_stopping = EarlyStopping(patience=PATIENCE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- 4. TRAINING LOOP ---
    print("\n" + "="*50)
    print("--- Starting RadX-KOA MobileNetV3 Model Training ---")
    print("="*50)
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_kappa, val_mcc = validate_epoch(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"Val Kappa: {val_kappa:.4f}, Val MCC: {val_mcc:.4f}")
        
        if early_stopping(val_loss, model):
            print(f"-> Early stopping triggered at epoch {epoch+1}! Restoring best model weights.")
            model.load_state_dict(early_stopping.best_model_state)
            break
            
    # Save the Final Champion Model
    best_model_path = os.path.join(SAVE_DIR, "radx_koa_MobileNetV3_model.pth")
    torch.save(model.state_dict(), best_model_path)
    print(f"\n-> Champion weights saved to: {best_model_path}")

    # ==========================================
    # --- 5. FINAL TEST SET EVALUATION ---
    # ==========================================
    print("\n" + "="*50)
    print("--- Evaluating ImageNet-based MobileNetV3 Model ---")
    print("="*50)
    
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    print("\n" + "="*50)
    print("      RADX-KOA: ImageNet-based MobileNetV3 FINAL TEST REPORT")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))
    print("-" * 50)
    print(f"Cohen's Kappa: {cohen_kappa_score(all_labels, all_preds):.4f}")
    print(f"MCC:           {matthews_corrcoef(all_labels, all_preds):.4f}")
    print("="*50 + "\n")

    # --- 6. PLOTTING BLOCK (PDF + 600 DPI PNG) ---
    plt.style.use('default') 
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - ImageNet-based MobileNetV3', fontsize=14)
    plt.ylabel('True Clinical Grade')
    plt.xlabel('Predicted Grade')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "MobileNetV3_Confusion_Matrix.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, "MobileNetV3_Confusion_Matrix.png"), format='png', dpi=600, bbox_inches='tight')
    plt.show()

    # ROC-AUC Curves
    y_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, color in zip(range(NUM_CLASSES), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC {CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curves of ImageNet-based MobileNetV3', fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "MobileNetV3_ROC_AUC.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, "MobileNetV3_ROC_AUC.png"), format='png', dpi=600, bbox_inches='tight')
    plt.show()

    # Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(NUM_CLASSES), colors):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        avg_precision = average_precision_score(y_bin[:, i], all_probs[:, i])
        plt.plot(recall, precision, color=color, lw=2, label=f'PR {CLASS_NAMES[i]} (AP = {avg_precision:.3f})')
    plt.xlabel('Recall (True Positive Rate)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title('Multi-Class PR Curves of MobileNetV3', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "MobileNetV3_PR_Curve.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, "MobileNetV3_PR_Curve.png"), format='png', dpi=600, bbox_inches='tight')
    plt.show()

    # --- 7. EXPLAINABLE AI (Grad-CAM) ---
    print("\nGenerating Grad-CAM Heatmaps for 5 Unseen Patients...")
    # Target the last convolutional block in MobileNetV3's feature extractor
    target_layer = model.features[-1]
    cam_extractor = GradCAM(model, target_layer)

    def show_cam_on_image(img_tensor, mask):
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * 0.5 + 0.5 
        img = np.clip(img, 0, 1)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    selected_indices = []
    for i in range(5):
        idx_options = np.where(all_labels == i)[0]
        if len(idx_options) > 0:
            selected_indices.append(idx_options[0])

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Explainable AI (Grad-CAM) Analysis: ImageNet-based MobileNetV3', fontsize=18)

    for col_idx, test_idx_in_subset in enumerate(selected_indices):
        img_tensor, true_label = test_dataset[test_idx_in_subset]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        heatmap, pred_label, _ = cam_extractor(input_tensor)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        overlay = show_cam_on_image(img_tensor, heatmap_resized)
        
        orig_img = img_tensor.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
        orig_img = np.clip(orig_img, 0, 1)
        
        axes[0, col_idx].imshow(orig_img)
        axes[0, col_idx].set_title(f"True: KL-{true_label}")
        axes[0, col_idx].axis('off')
        
        axes[1, col_idx].imshow(overlay)
        axes[1, col_idx].set_title(f"Pred: KL-{pred_label}")
        axes[1, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "MobileNetV3_GradCAM_Analysis.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, "MobileNetV3_GradCAM_Analysis.png"), format='png', dpi=600, bbox_inches='tight')
    plt.show()

    print("\nAll tasks complete!")