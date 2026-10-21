import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import label_binarize

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
import cv2
from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 10

# EXACT SAME OPTUNA-OBTAINED PARAMETERS
LR = 0.00041377355452253736
WEIGHT_DECAY = 5.555680375608746e-05
DROPOUT_RATE = 0.28

NUM_CLASSES = 5
CLASS_NAMES = ['KL-0', 'KL-1', 'KL-2', 'KL-3', 'KL-4']


# ============================================================
# 2. DATA PATHS
# ============================================================

TRAIN_DIR = r"C:\Users\vphar\KOA-Chen-Kaggle\archive\train"
VAL_DIR   = r"C:\Users\vphar\KOA-Chen-Kaggle\archive\val"
TEST_DIR  = r"C:\Users\vphar\KOA-Chen-Kaggle\archive\test"


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================
#
# IMPORTANT:
# Keep CE results separate from the original Focal Loss results.
#

SAVE_DIR = (
    r"C:\Users\vphar\KOA-Chen-Kaggle\PhD Objectives"
    r"\Re-Execution-Without-BalancedSet-New"
    r"\ImageNet\InceptionV3-CE-Ablation"
)

PLOT_DIR = SAVE_DIR

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 4. CLAHE PREPROCESSING
# ============================================================

class ApplyCLAHE(object):

    def __init__(
        self,
        clip_limit=2.0,
        tile_grid_size=(8, 8)
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):

        img_np = np.array(img.convert('L'))

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

        img_clahe = clahe.apply(img_np)

        return Image.fromarray(img_clahe).convert('RGB')


# ============================================================
# 5. CUSTOM INCEPTIONV3
# ============================================================

class CustomInceptionV3(nn.Module):

    def __init__(self, base_model):

        super(CustomInceptionV3, self).__init__()

        self.base_model = base_model

        num_features = self.base_model.fc.in_features

        self.base_model.fc = nn.Sequential(

            nn.Linear(
                num_features,
                1024
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                p=DROPOUT_RATE
            ),

            nn.Linear(
                1024,
                NUM_CLASSES
            )
        )

    def forward(self, x):

        outputs = self.base_model(x)

        if hasattr(outputs, 'logits'):
            return outputs.logits

        return outputs


# ============================================================
# 6. EARLY STOPPING
# ============================================================

class EarlyStopping:

    def __init__(
        self,
        patience=PATIENCE,
        min_delta=0.001
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):

        if val_loss < self.best_val_loss - self.min_delta:

            self.best_val_loss = val_loss
            self.epochs_no_improve = 0

            self.best_model_state = copy.deepcopy(
                model.state_dict()
            )

        else:

            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            return True

        return False


# ============================================================
# 7. GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(
            self.save_activation
        )

        target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def __call__(
        self,
        x,
        class_idx=None
    ):

        outputs = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(
                outputs,
                dim=1
            ).item()

        self.model.zero_grad()

        outputs[0, class_idx].backward(
            retain_graph=True
        )

        pooled_gradients = torch.mean(
            self.gradients,
            dim=[0, 2, 3]
        )

        activations = self.activations.detach()[0]

        for i in range(
            activations.shape[0]
        ):

            activations[i, :, :] *= (
                pooled_gradients[i]
            )

        heatmap = torch.mean(
            activations,
            dim=0
        ).squeeze().cpu().numpy()

        heatmap = np.maximum(
            heatmap,
            0
        )

        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        return (
            heatmap,
            class_idx,
            outputs
        )


# ============================================================
# 8. MODEL INITIALIZATION
# ============================================================

def get_model():

    # ImageNet initialization
    base = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1
    )

    # Disable auxiliary classifier after loading weights
    base.aux_logits = False
    base.AuxLogits = None

    return CustomInceptionV3(
        base
    ).to(device)


# ============================================================
# 9. TRAINING FUNCTION
# ============================================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for batch_idx, (
        inputs,
        labels
    ) in enumerate(pbar, 1):

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(
            outputs,
            1
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

        pbar.set_postfix({

            'loss':
                f"{running_loss / batch_idx:.4f}",

            'acc':
                f"{correct / total:.4f}"
        })

    return (
        running_loss / len(loader),
        correct / total
    )


# ============================================================
# 10. VALIDATION FUNCTION
# ============================================================

def validate_epoch(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_preds = []

    pbar = tqdm(
        loader,
        desc="Validating",
        leave=False
    )

    with torch.no_grad():

        for batch_idx, (
            inputs,
            labels
        ) in enumerate(pbar, 1):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            _, preds = torch.max(
                outputs,
                1
            )

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

            all_targets.extend(
                labels.cpu().numpy()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            pbar.set_postfix({

                'loss':
                    f"{running_loss / batch_idx:.4f}",

                'acc':
                    f"{correct / total:.4f}"
            })

    epoch_loss = (
        running_loss /
        len(loader)
    )

    epoch_acc = correct / total

    epoch_kappa = cohen_kappa_score(
        all_targets,
        all_preds
    )

    epoch_mcc = matthews_corrcoef(
        all_targets,
        all_preds
    )

    return (
        epoch_loss,
        epoch_acc,
        epoch_kappa,
        epoch_mcc
    )


# ============================================================
# 11. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("INCEPTIONV3 IMAGENET - CROSS ENTROPY ABLATION")
    print("=" * 70)

    print(
        f"\nDevice: {device}"
    )

    print(
        "\nControlled ablation:"
    )

    print(
        "  Backbone       : InceptionV3"
    )

    print(
        "  Initialization : ImageNet"
    )

    print(
        "  Loss           : Cross Entropy"
    )

    print(
        "  All other settings remain identical to Focal Loss experiment."
    )


    # ========================================================
    # 12. TRANSFORMS
    # ========================================================

    train_transform = transforms.Compose([

        ApplyCLAHE(
            clip_limit=2.0
        ),

        transforms.Resize(
            (299, 299)
        ),

        transforms.RandomRotation(
            15
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomResizedCrop(
            (299, 299),
            scale=(0.85, 1.0)
        ),

        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


    val_test_transform = transforms.Compose([

        ApplyCLAHE(
            clip_limit=2.0
        ),

        transforms.Resize(
            (299, 299)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


    # ========================================================
    # 13. DATASETS
    # ========================================================

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=val_test_transform
    )


    # ========================================================
    # 14. DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


    print(
        "\nDataset sizes:"
    )

    print(
        f"Training     : {len(train_dataset)}"
    )

    print(
        f"Validation   : {len(val_dataset)}"
    )

    print(
        f"Independent Test : {len(test_dataset)}"
    )


    # ========================================================
    # 15. MODEL
    # ========================================================

    model = get_model()


    # ========================================================
    # 16. OPTIMIZER
    # ========================================================

    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


    # ========================================================
    # 17. CROSS-ENTROPY LOSS
    # ========================================================

    # ONLY MAJOR CHANGE FROM THE FOCAL-LOSS EXPERIMENT

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # 18. EARLY STOPPING
    # ========================================================

    early_stopping = EarlyStopping(
        patience=PATIENCE
    )


    # ========================================================
    # 19. LEARNING-RATE SCHEDULER
    # ========================================================

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode='min',

        factor=0.5,

        patience=3
    )


    # ========================================================
    # 20. TRAINING
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING CROSS-ENTROPY TRAINING")
    print("=" * 70)


    for epoch in range(
        NUM_EPOCHS
    ):

        train_loss, train_acc = train_epoch(

            model,

            train_loader,

            criterion,

            optimizer
        )


        val_loss, val_acc, val_kappa, val_mcc = validate_epoch(

            model,

            val_loader,

            criterion
        )


        scheduler.step(
            val_loss
        )


        print(

            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "

            f"Train Loss: {train_loss:.4f} | "

            f"Train Acc: {train_acc:.4f} | "

            f"Val Loss: {val_loss:.4f} | "

            f"Val Acc: {val_acc:.4f} | "

            f"Val Kappa: {val_kappa:.4f} | "

            f"Val MCC: {val_mcc:.4f}"
        )


        if early_stopping(
            val_loss,
            model
        ):

            print(
                f"\nEarly stopping triggered at epoch "
                f"{epoch + 1}"
            )

            model.load_state_dict(
                early_stopping.best_model_state
            )

            break


    # ========================================================
    # 21. SAVE BEST CROSS-ENTROPY MODEL
    # ========================================================

    best_model_path = os.path.join(

        SAVE_DIR,

        "InceptionV3_ImageNet_CE_model.pth"
    )

    torch.save(
        model.state_dict(),
        best_model_path
    )


    print(
        f"\nBest Cross-Entropy model saved to:\n"
        f"{best_model_path}"
    )


    # ========================================================
    # 22. FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION - CROSS ENTROPY")
    print("=" * 70)


    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []


    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(
                inputs
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )

            _, preds = torch.max(
                outputs,
                1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )


    all_labels = np.array(
        all_labels
    )

    all_preds = np.array(
        all_preds
    )

    all_probs = np.array(
        all_probs
    )


    # ========================================================
    # 23. CLASSIFICATION METRICS
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    macro_precision = precision_score(
        all_labels,
        all_preds,
        average='macro',
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_preds,
        average='macro',
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average='macro',
        zero_division=0
    )

    kappa = cohen_kappa_score(
        all_labels,
        all_preds
    )

    mcc = matthews_corrcoef(
        all_labels,
        all_preds
    )


    # ========================================================
    # 24. MULTI-CLASS MACRO AUC
    # ========================================================

    y_bin = label_binarize(
        all_labels,
        classes=np.arange(NUM_CLASSES)
    )


    macro_auc = roc_auc_score(

        all_labels,

        all_probs,

        multi_class='ovr',

        average='macro'
    )


    micro_auc = roc_auc_score(

        y_bin,

        all_probs,

        average='micro'
    )


    # ========================================================
    # 25. PRINT FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "INCEPTIONV3 IMAGENET - CROSS ENTROPY"
    )

    print(
        "FINAL TEST RESULTS"
    )

    print("=" * 70)


    print(
        classification_report(

            all_labels,

            all_preds,

            target_names=CLASS_NAMES,

            digits=4,

            zero_division=0
        )
    )


    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Macro Precision: {macro_precision:.4f}"
    )

    print(
        f"Macro Recall   : {macro_recall:.4f}"
    )

    print(
        f"Macro F1       : {macro_f1:.4f}"
    )

    print(
        f"Macro AUC      : {macro_auc:.4f}"
    )

    print(
        f"Micro AUC      : {micro_auc:.4f}"
    )

    print(
        f"Cohen Kappa    : {kappa:.4f}"
    )

    print(
        f"MCC            : {mcc:.4f}"
    )

    print("=" * 70)


    # ========================================================
    # 26. SAVE NUMERICAL RESULTS
    # ========================================================

    results_file = os.path.join(

        SAVE_DIR,

        "InceptionV3_ImageNet_CE_results.txt"
    )


    with open(
        results_file,
        "w"
    ) as f:

        f.write(
            "InceptionV3 ImageNet - "
            "Cross Entropy Ablation\n"
        )

        f.write(
            "=" * 60 + "\n"
        )

        f.write(
            f"Loss: CrossEntropyLoss\n"
        )

        f.write(
            f"Learning Rate: {LR}\n"
        )

        f.write(
            f"Weight Decay: {WEIGHT_DECAY}\n"
        )

        f.write(
            f"Dropout: {DROPOUT_RATE}\n"
        )

        f.write(
            f"Batch Size: {BATCH_SIZE}\n"
        )

        f.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        f.write(
            f"Macro Precision: {macro_precision:.6f}\n"
        )

        f.write(
            f"Macro Recall: {macro_recall:.6f}\n"
        )

        f.write(
            f"Macro F1: {macro_f1:.6f}\n"
        )

        f.write(
            f"Macro AUC: {macro_auc:.6f}\n"
        )

        f.write(
            f"Micro AUC: {micro_auc:.6f}\n"
        )

        f.write(
            f"Cohen Kappa: {kappa:.6f}\n"
        )

        f.write(
            f"MCC: {mcc:.6f}\n"
        )


    # ========================================================
    # 27. CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        all_labels,
        all_preds
    )


    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(

        cm,

        annot=True,

        fmt='d',

        cmap='Blues',

        xticklabels=CLASS_NAMES,

        yticklabels=CLASS_NAMES
    )

    plt.title(
        'InceptionV3 ImageNet - Cross Entropy'
    )

    plt.ylabel(
        'True Clinical Grade'
    )

    plt.xlabel(
        'Predicted Grade'
    )

    plt.tight_layout()


    plt.savefig(

        os.path.join(
            PLOT_DIR,
            "InceptionV3_CE_Confusion_Matrix.pdf"
        ),

        format='pdf',

        bbox_inches='tight'
    )


    plt.savefig(

        os.path.join(
            PLOT_DIR,
            "InceptionV3_CE_Confusion_Matrix.png"
        ),

        dpi=600,

        bbox_inches='tight'
    )

    plt.close()


    # ========================================================
    # 28. ROC CURVES
    # ========================================================

    plt.figure(
        figsize=(10, 8)
    )


    colors = [
        'blue',
        'red',
        'green',
        'purple',
        'orange'
    ]


    for i, color in zip(
        range(NUM_CLASSES),
        colors
    ):

        fpr, tpr, _ = roc_curve(

            y_bin[:, i],

            all_probs[:, i]
        )


        roc_auc = auc(
            fpr,
            tpr
        )


        plt.plot(

            fpr,

            tpr,

            color=color,

            lw=2,

            label=(
                f"ROC {CLASS_NAMES[i]} "
                f"(AUC={roc_auc:.3f})"
            )
        )


    plt.plot(
        [0, 1],
        [0, 1],
        'k--'
    )


    plt.xlabel(
        'False Positive Rate'
    )

    plt.ylabel(
        'True Positive Rate'
    )

    plt.title(
        'InceptionV3 ImageNet - Cross Entropy ROC'
    )

    plt.legend(
        loc='lower right'
    )

    plt.tight_layout()


    plt.savefig(

        os.path.join(
            PLOT_DIR,
            "InceptionV3_CE_ROC_AUC.pdf"
        ),

        format='pdf',

        bbox_inches='tight'
    )


    plt.savefig(

        os.path.join(
            PLOT_DIR,
            "InceptionV3_CE_ROC_AUC.png"
        ),

        dpi=600,

        bbox_inches='tight'
    )

    plt.close()


    # ========================================================
    # 29. PRECISION-RECALL CURVES
    # ========================================================

    plt.figure(
        figsize=(10, 8)
    )


    for i, color in zip(
        range(NUM_CLASSES),
        colors
    ):

        precision, recall, _ = (
            precision_recall_curve(

                y_bin[:, i],

                all_probs[:, i]
            )
        )


        avg_precision = (
            average_precision_score(

                y_bin[:, i],

                all_probs[:, i]
            )
        )


        plt.plot(

            recall,

            precision,

            color=color,

            lw=2,

            label=(
                f"PR {CLASS_NAMES[i]} "
                f"(AP={avg_precision:.3f})"
            )
        )


    plt.xlabel(
        'Recall'
    )

    plt.ylabel(
        'Precision'
    )

    plt.title(
        'InceptionV3 ImageNet - Cross Entropy PR Curves'
    )

    plt.legend(
        loc='lower left'
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(

        os.path.join(
            PLOT_DIR,
            "InceptionV3_CE_PR_Curve.pdf"
        ),

        format='pdf',

        bbox_inches='tight'
    )


    plt.savefig(

        os.path.join(
            PLOT_DIR,
            "InceptionV3_CE_PR_Curve.png"
        ),

        dpi=600,

        bbox_inches='tight'
    )

    plt.close()


    # ========================================================
    # 30. GRAD-CAM
    # ========================================================

    print(
        "\nGenerating Grad-CAM examples..."
    )


    target_layer = (
        model.base_model.Mixed_7c
    )


    cam_extractor = GradCAM(

        model,

        target_layer
    )


    def show_cam_on_image(
        img_tensor,
        mask
    ):

        img = (
            img_tensor.cpu()
            .numpy()
            .transpose(1, 2, 0)
        )


        img = (
            img * 0.5 + 0.5
        )


        img = np.clip(
            img,
            0,
            1
        )


        heatmap = cv2.applyColorMap(

            np.uint8(
                255 * mask
            ),

            cv2.COLORMAP_JET
        )


        heatmap = (
            np.float32(
                heatmap
            ) / 255
        )


        cam = (
            heatmap +
            np.float32(img)
        )


        cam = (
            cam /
            np.max(cam)
        )


        return np.uint8(
            255 * cam
        )


    # Select one example per true class
    selected_indices = []


    for i in range(NUM_CLASSES):

        idx_options = np.where(
            all_labels == i
        )[0]

        if len(idx_options) > 0:

            selected_indices.append(
                idx_options[0]
            )


    fig, axes = plt.subplots(
        2,
        5,
        figsize=(20, 8)
    )


    fig.suptitle(
        'Grad-CAM Analysis: '
        'InceptionV3 ImageNet - Cross Entropy',
        fontsize=18
    )


    for col_idx, test_idx in enumerate(
        selected_indices
    ):

        img_tensor, true_label = (
            test_dataset[test_idx]
        )


        input_tensor = (
            img_tensor
            .unsqueeze(0)
            .to(device)
        )


        heatmap, pred_label, _ = (
            cam_extractor(
                input_tensor
            )
        )


        heatmap_resized = cv2.resize(

            heatmap,

            (299, 299)
        )


        overlay = show_cam_on_image(

            img_tensor,

            heatmap_resized
        )


        orig_img = (

            img_tensor.cpu()
            .numpy()
            .transpose(1, 2, 0)
        )


        orig_img = (
            orig_img * 0.5 + 0.5
        )


        orig_img = np.clip(
            orig_img,
            0,
            1
        )


        axes[
            0,
            col_idx
        ].imshow(
            orig_img
        )

        axes[
            0,
            col_idx
        ].set_title(
            f"True: KL-{true_label}"
        )

        axes[
            0,
            col_idx
        ].axis('off')


        axes[
            1,
            col_idx
        ].imshow(
            overlay
        )

        axes[
            1,
            col_idx
        ].set_title(
            f"Pred: KL-{pred_label}"
        )

        axes[
            1,
            col_idx
        ].axis('off')


    # Hide unused panels
    for j in range(
        len(selected_indices),
        NUM_CLASSES
    ):

        axes[
            0,
            j
        ].axis('off')

        axes[
            1,
            j
        ].axis('off')


    plt.tight_layout()


    plt.savefig(

        os.path.join(
            PLOT_DIR,
            "InceptionV3_CE_GradCAM.pdf"
        ),

        format='pdf',

        bbox_inches='tight'
    )


    plt.savefig(

        os.path.join(
            PLOT_DIR,
            "InceptionV3_CE_GradCAM.png"
        ),

        dpi=600,

        bbox_inches='tight'
    )


    plt.close()


    # ========================================================
    # 31. COMPLETE
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "CROSS-ENTROPY ABLATION COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nResults saved in:\n{SAVE_DIR}"
    )

    print(
        "\nCompare this output directly against "
        "your original Focal Loss run."
    )

    print(
        "\nIMPORTANT: Do not use the test set to select "
        "between CE and Focal Loss."
    )

    print(
        "The comparison should be reported as a "
        "post-development loss-function ablation."
    )

    print("=" * 70)