# ============================================================
# EdgeX-KOA (Corrected Directory Mapping Version)
# Leakage-Controlled Edge Calibration + Hold-Out Evaluation
# + Grad-CAM++ Interpretability Analysis
# ============================================================

import os
import copy
import random
import warnings
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    accuracy_score
)
from sklearn.preprocessing import label_binarize

from skimage.exposure import match_histograms

import matplotlib.pyplot as plt

# Grad-CAM++
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


warnings.filterwarnings("ignore")


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 2. CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
NUM_CLASSES = 5

CLASS_NAMES = [
    "KL-0",
    "KL-1",
    "KL-2",
    "KL-3",
    "KL-4"
]

CALIBRATION_RATIO = 0.20
TEST_RATIO = 0.80

CALIBRATION_EPOCHS = 5
CALIBRATION_LR = 1e-3

NUM_WORKERS = 0

# Number of Grad-CAM++ examples per category
MAX_XAI_PER_CATEGORY = 20

# ============================================================
# 3. DATASET PATHS
# ============================================================

EXPERT_1_DIR = (
    r"C:\Users\vphar\KOA-Chen-Kaggle\archive\Test-Newset"
    r"\Digital Knee X-ray Images\Digital Knee X-ray Images"
    r"\Knee X-ray Images\MedicalExpert-I\MedicalExpert-I"
)

EXPERT_2_DIR = (
    r"C:\Users\vphar\KOA-Chen-Kaggle\archive\Test-Newset"
    r"\Digital Knee X-ray Images\Digital Knee X-ray Images"
    r"\Knee X-ray Images\MedicalExpert-II\MedicalExpert-II"
)

# ============================================================
# 4. MODEL WEIGHTS
# ============================================================

IMAGENET_WEIGHTS = (
    r"C:\Users\vphar\KOA-Chen-Kaggle\PhD Objectives"
    r"\Re-Execution-Without-BalancedSet-New\ImageNet"
    r"\InceptionV3-Img-Files\radx_koa_inceptionv3_rad_model.pth"
)

RADIMAGENET_WEIGHTS = (
    r"C:\Users\vphar\KOA-Chen-Kaggle\PhD Objectives"
    r"\Re-Execution-Without-BalancedSet-New"
    r"\InceptionV3-Rad-Files"
    r"\radx_koa_inceptionv3_rad_model.pth"
)

# ============================================================
# 5. OUTPUT DIRECTORY
# ============================================================

OUTPUT_ROOT = (
    r"C:\Users\vphar\KOA-Chen-Kaggle"
    r"\EdgeX-KOA_ExternalValidation_XAI"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ============================================================
# 6. OAI REFERENCE IMAGE
# ============================================================

OAI_REFERENCE_PATH = (
    r"C:\Users\vphar\KOA-Chen-Kaggle\archive"
    r"\train\0\9001695L.png"
)


# ============================================================
# 7. IMAGE PREPROCESSING
# ============================================================

class ApplyCLAHE:

    def __init__(
        self,
        clip_limit=2.0,
        tile_grid_size=(8, 8)
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):

        import cv2

        img_np = np.array(
            img.convert("L")
        )

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

        img_clahe = clahe.apply(img_np)

        return Image.fromarray(
            img_clahe
        ).convert("RGB")


class HistogramMatcher:

    def __init__(self, reference_image_path):

        ref_img = Image.open(
            reference_image_path
        ).convert("L")

        self.reference_arr = np.array(ref_img)

    def __call__(self, img):

        img_arr = np.array(
            img.convert("L")
        )

        matched_arr = match_histograms(
            img_arr,
            self.reference_arr
        )

        matched_arr = np.clip(
            matched_arr,
            0,
            255
        ).astype(np.uint8)

        return Image.fromarray(
            matched_arr
        ).convert("RGB")


# ============================================================
# 8. TRANSFORM
# ============================================================

VAL_TRANSFORM = transforms.Compose([

    HistogramMatcher(
        OAI_REFERENCE_PATH
    ),

    ApplyCLAHE(
        clip_limit=2.0,
        tile_grid_size=(8, 8)
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


# ============================================================
# 9. IMAGE RECORD COLLECTION (With Robust Folder Mapping)
# ============================================================

def collect_image_records(root_dir):

    records = []

    # Map actual dataset folder naming conventions to KL classes and indices
    folder_mapping = {
        "0Normal": ("KL-0", 0),
        "1Doubtful": ("KL-1", 1),
        "2Mild": ("KL-2", 2),
        "3Moderate": ("KL-3", 3),
        "4Severe": ("KL-4", 4),
        # Fallbacks if standard KL folders happen to be present
        "KL-0": ("KL-0", 0),
        "KL-1": ("KL-1", 1),
        "KL-2": ("KL-2", 2),
        "KL-3": ("KL-3", 3),
        "KL-4": ("KL-4", 4)
    }

    if not os.path.exists(root_dir):
        print(f"ERROR: Root directory does not exist: {root_dir}")
        return pd.DataFrame(records)

    subdirs = os.listdir(root_dir)
    for folder_name in subdirs:
        if folder_name in folder_mapping:
            class_name, label_idx = folder_mapping[folder_name]
            class_dir = os.path.join(root_dir, folder_name)

            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    path = os.path.join(class_dir, filename)
                    records.append({
                        "path": path,
                        "filename": filename,
                        "class_name": class_name,
                        "label": label_idx
                    })

    if len(records) == 0:
        print(f"WARNING: No images found in {root_dir}. Please check subfolders (found: {subdirs}).")

    return pd.DataFrame(records)


# ============================================================
# 10. DATASET CLASS
# ============================================================

class ImageRecordDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["path"]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])

        return (
            image,
            label,
            row["filename"],
            row["path"]
        )


# ============================================================
# 11. CREATE FIXED STRATIFIED SPLIT
# ============================================================

def create_stratified_split(df):

    calibration_df, holdout_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        train_size=CALIBRATION_RATIO,
        stratify=df["label"],
        random_state=SEED
    )

    calibration_df = calibration_df.reset_index(
        drop=True
    )

    holdout_df = holdout_df.reset_index(
        drop=True
    )

    return calibration_df, holdout_df


# ============================================================
# 12. MODEL
# ============================================================

class CustomInceptionV3(nn.Module):

    def __init__(self):

        super().__init__()

        self.base_model = models.inception_v3(
            weights=None,
            init_weights=False,
            aux_logits=False
        )

        self.base_model.AuxLogits = None

        num_features = (
            self.base_model.fc.in_features
        )

        self.base_model.fc = nn.Sequential(

            nn.Linear(
                num_features,
                1024
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.28
            ),

            nn.Linear(
                1024,
                NUM_CLASSES
            )
        )

    def forward(self, x):

        outputs = self.base_model(x)

        if hasattr(
            outputs,
            "logits"
        ):

            return outputs.logits

        return outputs


def get_model():

    model = CustomInceptionV3()

    return model.to(DEVICE)


# ============================================================
# 13. LOAD WEIGHTS
# ============================================================

def load_weights(
    model,
    weight_path
):

    print(
        f"\nLoading weights:\n{weight_path}"
    )

    checkpoint = torch.load(
        weight_path,
        map_location=DEVICE,
        weights_only=True
    )

    missing, unexpected = (
        model.load_state_dict(
            checkpoint,
            strict=False
        )
    )

    if missing:
        print(
            "\nMissing keys:"
        )

        for k in missing[:10]:
            print(" ", k)

    if unexpected:
        print(
            "\nUnexpected keys:"
        )

        for k in unexpected[:10]:
            print(" ", k)

    return model


# ============================================================
# 14. EDGE CALIBRATION
# ============================================================

def calibrate_on_edge(
    model,
    calibration_df,
    transform
):

    print(
        "\n========================================"
    )

    print(
        "EDGE CALIBRATION"
    )

    print(
        "Calibration images:",
        len(calibration_df)
    )

    print(
        "Calibration epochs:",
        CALIBRATION_EPOCHS
    )

    print(
        "========================================"
    )

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze final classifier only
    for param in (
        model.base_model.fc.parameters()
    ):
        param.requires_grad = True

    dataset = ImageRecordDataset(
        calibration_df,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    optimizer = torch.optim.SGD(
    model.base_model.fc.parameters(),
    lr=CALIBRATION_LR, 
    weight_decay=5.6e-5 
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(
        CALIBRATION_EPOCHS
    ):

        running_loss = 0.0
        correct = 0
        total = 0

        for (
            inputs,
            labels,
            _,
            _
        ) in loader:

            inputs = inputs.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * inputs.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        epoch_loss = (
            running_loss / total
        )

        epoch_acc = (
            correct / total
        )

        print(
            f"Epoch {epoch + 1}/"
            f"{CALIBRATION_EPOCHS} | "
            f"Loss={epoch_loss:.4f} | "
            f"Accuracy={epoch_acc:.4f}"
        )

    print(
        "\nEdge calibration completed."
    )

    return model


# ============================================================
# 15. EVALUATION
# ============================================================

def evaluate_model(
    model,
    holdout_df,
    transform,
    model_name
):

    dataset = ImageRecordDataset(
        holdout_df,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    all_filenames = []
    all_paths = []

    with torch.no_grad():

        for (
            inputs,
            labels,
            filenames,
            paths
        ) in loader:

            inputs = inputs.to(
                DEVICE
            )

            outputs = model(
                inputs
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )

            preds = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_filenames.extend(
                filenames
            )

            all_paths.extend(
                paths
            )

    labels = np.array(
        all_labels
    )

    preds = np.array(
        all_preds
    )

    probs = np.array(
        all_probs
    )

    accuracy = accuracy_score(
        labels,
        preds
    )

    kappa = cohen_kappa_score(
        labels,
        preds
    )

    mcc = matthews_corrcoef(
        labels,
        preds
    )

    y_bin = label_binarize(
        labels,
        classes=np.arange(
            NUM_CLASSES
        )
    )

    try:

        macro_auc = roc_auc_score(
            y_bin,
            probs,
            multi_class="ovr",
            average="macro"
        )

    except Exception:

        macro_auc = np.nan

    report = classification_report(
        labels,
        preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print(
        "\n=========================================="
    )

    print(
        f"{model_name} — HOLD-OUT TEST"
    )

    print(
        "=========================================="
    )

    print(
        report
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Kappa    : {kappa:.4f}"
    )

    print(
        f"MCC      : {mcc:.4f}"
    )

    print(
        f"Macro AUC: {macro_auc:.4f}"
    )

    cm = confusion_matrix(
        labels,
        preds
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    results_df = pd.DataFrame({

        "filename":
            all_filenames,

        "path":
            all_paths,

        "true_label":
            labels,

        "true_class":
            [
                CLASS_NAMES[x]
                for x in labels
            ],

        "pred_label":
            preds,

        "pred_class":
            [
                CLASS_NAMES[x]
                for x in preds
            ],

        "confidence":
            probs.max(axis=1),

        "correct":
            labels == preds
    })

    return (
        results_df,
        {
            "accuracy": accuracy,
            "kappa": kappa,
            "mcc": mcc,
            "macro_auc": macro_auc,
            "confusion_matrix": cm
        }
    )


# ============================================================
# 16. EXPERT-II LABEL MATCHING
# ============================================================

def build_filename_label_map(
    expert2_dir
):

    expert2_df = collect_image_records(
        expert2_dir
    )

    return dict(
        zip(
            expert2_df["filename"],
            expert2_df["label"]
        )
    )


def evaluate_against_expert2(
    results_df,
    expert2_dir,
    model_name
):

    expert2_map = (
        build_filename_label_map(
            expert2_dir
        )
    )

    results_df = results_df.copy()

    results_df[
        "expert2_label"
    ] = results_df[
        "filename"
    ].map(expert2_map)

    valid = (
        results_df[
            "expert2_label"
        ].notna()
    )

    if valid.sum() == 0:

        print(
            "\nNo matching Expert-II "
            "filenames found."
        )

        return results_df

    expert2_labels = (
        results_df.loc[
            valid,
            "expert2_label"
        ].astype(int).values
    )

    predictions = (
        results_df.loc[
            valid,
            "pred_label"
        ].values
    )

    expert2_acc = accuracy_score(
        expert2_labels,
        predictions
    )

    expert2_kappa = (
        cohen_kappa_score(
            expert2_labels,
            predictions
        )
    )

    print(
        "\n=========================================="
    )

    print(
        f"{model_name} — EXPERT-II "
        "SECOND-READER ANALYSIS"
    )

    print(
        "=========================================="
    )

    print(
        f"Matched hold-out images: "
        f"{valid.sum()}"
    )

    print(
        f"Accuracy vs Expert-II: "
        f"{expert2_acc:.4f}"
    )

    print(
        f"Kappa vs Expert-II: "
        f"{expert2_kappa:.4f}"
    )

    return results_df


# ============================================================
# 17. GRAD-CAM++ TARGET LAYER
# ============================================================

def get_gradcam_target_layer(
    model
):

    return model.base_model.Mixed_7c


# ============================================================
# 18. DENORMALIZATION
# ============================================================

def tensor_to_display_image(
    tensor
):

    image = tensor.detach().cpu().numpy()

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    mean = np.array(
        [0.5, 0.5, 0.5]
    )

    std = np.array(
        [0.5, 0.5, 0.5]
    )

    image = (
        image * std
    ) + mean

    image = np.clip(
        image,
        0,
        1
    )

    return image.astype(
        np.float32
    )


# ============================================================
# 19. GRAD-CAM++ QUANTITATIVE METRICS
# ============================================================

def calculate_cam_statistics(
    cam
):

    cam = np.asarray(
        cam,
        dtype=np.float32
    )

    cam = np.clip(
        cam,
        0,
        1
    )

    eps = 1e-8

    mean_activation = float(
        cam.mean()
    )

    max_activation = float(
        cam.max()
    )

    active_fraction = float(
        np.mean(
            cam >= 0.5
        )
    )

    flat = cam.flatten()

    probability = (
        flat + eps
    ) / (
        flat.sum() + eps * len(flat)
    )

    entropy = float(
        -np.sum(
            probability
            * np.log(
                probability + eps
            )
        )
    )

    normalized_entropy = (
        entropy /
        np.log(len(flat))
    )

    return {
        "cam_mean": mean_activation,
        "cam_max": max_activation,
        "cam_active_fraction": active_fraction,
        "cam_normalized_entropy":
            normalized_entropy
    }


# ============================================================
# 20. SAVE GRAD-CAM++ IMAGE
# ============================================================

def save_gradcam_image(
    image,
    cam,
    output_path,
    title
):

    visualization = show_cam_on_image(
        image,
        cam,
        use_rgb=True
    )

    plt.figure(
        figsize=(5, 5)
    )

    plt.imshow(
        visualization
    )

    plt.axis(
        "off"
    )

    plt.title(
        title,
        fontsize=9
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 21. GRAD-CAM++ ANALYSIS (Fixed for Frozen Backbones)
# ============================================================

def generate_gradcam_analysis(
    model,
    results_df,
    model_name,
    transform
):

    print(
        "\n=========================================="
    )

    print(
        f"Grad-CAM++ Analysis: {model_name}"
    )

    print(
        "=========================================="
    )

    model.eval()

    # Temporarily enable requires_grad so Grad-CAM++ can compute gradients
    for param in model.parameters():
        param.requires_grad = True

    target_layer = (
        get_gradcam_target_layer(
            model
        )
    )

    cam_algorithm = GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer]
    )

    model_output_dir = os.path.join(
        OUTPUT_ROOT,
        model_name.replace(
            " ",
            "_"
        ),
        "GradCAMPlusPlus"
    )

    os.makedirs(
        model_output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Category tracking
    # --------------------------------------------------------

    category_counts = {}

    xai_records = []

    # --------------------------------------------------------
    # Iterate over hold-out predictions
    # --------------------------------------------------------

    for _, row in results_df.iterrows():

        true_class = row[
            "true_class"
        ]

        pred_class = row[
            "pred_class"
        ]

        correctness = (
            "Correct"
            if row["correct"]
            else "Incorrect"
        )

        category_key = (
            f"{true_class}_{correctness}"
        )

        current_count = (
            category_counts.get(
                category_key,
                0
            )
        )

        # Limit number of visualizations
        if current_count >= (
            MAX_XAI_PER_CATEGORY
        ):
            continue

        category_counts[
            category_key
        ] = current_count + 1

        # ----------------------------------------------------
        # Folder structure
        # ----------------------------------------------------

        save_dir = os.path.join(
            model_output_dir,
            true_class,
            correctness
        )

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image_pil = Image.open(
            row["path"]
        ).convert("RGB")

        input_tensor = transform(
            image_pil
        ).unsqueeze(0)

        input_tensor = (
            input_tensor.to(
                DEVICE
            )
        )

        # ----------------------------------------------------
        # Forward pass (with gradients enabled for CAM)
        # ----------------------------------------------------

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = (
            output.argmax(
                dim=1
            ).item()
        )

        confidence = (
            probabilities[
                0,
                predicted_class
            ].item()
        )

        # ----------------------------------------------------
        # Grad-CAM++ target
        # ----------------------------------------------------

        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]

        grayscale_cam = cam_algorithm(
            input_tensor=input_tensor,
            targets=targets
        )[0]

        # ----------------------------------------------------
        # Convert input to display image
        # ----------------------------------------------------

        display_image = (
            tensor_to_display_image(
                input_tensor[0]
            )
        )

        # ----------------------------------------------------
        # Quantitative CAM statistics
        # ----------------------------------------------------

        cam_stats = (
            calculate_cam_statistics(
                grayscale_cam
            )
        )

        # ----------------------------------------------------
        # Save visualization
        # ----------------------------------------------------

        filename_base = os.path.splitext(
            row["filename"]
        )[0]

        output_filename = (
            f"{filename_base}"
            f"__True-{true_class}"
            f"__Pred-{pred_class}"
            f"__{correctness}"
            f"__conf-{confidence:.3f}.png"
        )

        output_path = os.path.join(
            save_dir,
            output_filename
        )

        title = (
            f"{model_name}\n"
            f"True={true_class} | "
            f"Pred={pred_class} | "
            f"Confidence={confidence:.3f}"
        )

        save_gradcam_image(
            display_image,
            grayscale_cam,
            output_path,
            title
        )

        # ----------------------------------------------------
        # Store quantitative record
        # ----------------------------------------------------

        xai_records.append({

            "model":
                model_name,

            "filename":
                row["filename"],

            "true_class":
                true_class,

            "predicted_class":
                pred_class,

            "correct":
                bool(row["correct"]),

            "confidence":
                confidence,

            "cam_mean":
                cam_stats[
                    "cam_mean"
                ],

            "cam_max":
                cam_stats[
                    "cam_max"
                ],

            "cam_active_fraction":
                cam_stats[
                    "cam_active_fraction"
                ],

            "cam_normalized_entropy":
                cam_stats[
                    "cam_normalized_entropy"
                ],

            "output_path":
                output_path
        })

    # --------------------------------------------------------
    # Save XAI CSV
    # --------------------------------------------------------

    xai_df = pd.DataFrame(
        xai_records
    )

    csv_path = os.path.join(
        model_output_dir,
        "GradCAMPlusPlus_Quantitative_Summary.csv"
    )

    xai_df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nSaved Grad-CAM++ summary:\n"
        f"{csv_path}"
    )

    return xai_df


# ============================================================
# 22. XAI GROUP SUMMARY
# ============================================================

def summarize_xai(
    xai_df,
    model_name
):

    if xai_df.empty:
        return

    summary = (
        xai_df
        .groupby(
            [
                "true_class",
                "correct"
            ]
        )
        .agg({

            "confidence":
                ["mean", "std"],

            "cam_mean":
                ["mean", "std"],

            "cam_active_fraction":
                ["mean", "std"],

            "cam_normalized_entropy":
                ["mean", "std"]
        })
        .reset_index()
    )

    summary.columns = [
        "_".join(
            [str(x) for x in col]
        ).strip("_")
        for col in summary.columns
    ]

    output_dir = os.path.join(
        OUTPUT_ROOT,
        model_name.replace(
            " ",
            "_"
        ),
        "GradCAMPlusPlus"
    )

    output_path = os.path.join(
        output_dir,
        "GradCAMPlusPlus_Group_Summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False
    )

    print(
        "\nQuantitative XAI group summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nSaved:\n{output_path}"
    )


# ============================================================
# 23. SAVE MAIN EVALUATION RESULTS
# ============================================================

def save_evaluation_results(
    results_df,
    metrics,
    model_name
):

    model_dir = os.path.join(
        OUTPUT_ROOT,
        model_name.replace(
            " ",
            "_"
        )
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    results_path = os.path.join(
        model_dir,
        "Holdout_Predictions.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    metrics_path = os.path.join(
        model_dir,
        "Evaluation_Metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Model: {model_name}\n"
        )

        f.write(
            f"Accuracy: "
            f"{metrics['accuracy']:.6f}\n"
        )

        f.write(
            f"Cohen's Kappa: "
            f"{metrics['kappa']:.6f}\n"
        )

        f.write(
            f"MCC: "
            f"{metrics['mcc']:.6f}\n"
        )

        f.write(
            f"Macro AUC: "
            f"{metrics['macro_auc']:.6f}\n"
        )

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            np.array2string(
                metrics[
                    "confusion_matrix"
                ]
            )
        )

    print(
        f"\nSaved evaluation results:"
        f"\n{results_path}"
        f"\n{metrics_path}"
    )


# ============================================================
# 24. MAIN PIPELINE
# ============================================================

def main():

    print(
        "\n===================================================="
    )

    print(
        "EdgeX-KOA Leakage-Controlled "
        "Evaluation + Grad-CAM++"
    )

    print(
        "===================================================="
    )

    print(
        f"Device: {DEVICE}"
    )

    expert1_df = collect_image_records(
        EXPERT_1_DIR
    )

    print(
        f"\nTotal Expert-I images: "
        f"{len(expert1_df)}"
    )

    if len(expert1_df) == 0:
        print("ERROR: No images loaded. Please verify the dataset path.")
        return

    print(
        "\nExpert-I class distribution:"
    )

    print(
        expert1_df[
            "class_name"
        ].value_counts()
        .sort_index()
    )

    calibration_df, holdout_df = (
        create_stratified_split(
            expert1_df
        )
    )

    print(
        "\n=========================================="
    )

    print(
        "DATA SPLIT"
    )

    print(
        "=========================================="
    )

    print(
        f"Calibration: "
        f"{len(calibration_df)} "
        f"({100 * CALIBRATION_RATIO:.1f}%)"
    )

    print(
        f"Hold-out: "
        f"{len(holdout_df)} "
        f"({100 * TEST_RATIO:.1f}%)"
    )

    print(
        "\nCalibration distribution:"
    )

    print(
        calibration_df[
            "class_name"
        ].value_counts()
        .sort_index()
    )

    print(
        "\nHold-out distribution:"
    )

    print(
        holdout_df[
            "class_name"
        ].value_counts()
        .sort_index()
    )

    calibration_df.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "Calibration_Split.csv"
        ),
        index=False
    )

    holdout_df.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "Holdout_Split.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Run ImageNet
    # --------------------------------------------------------

    print(
        "\n\n########################################"
    )

    print(
        "IMAGE-NET INITIALIZATION"
    )

    print(
        "########################################"
    )

    model_img = get_model()

    model_img = load_weights(
        model_img,
        IMAGENET_WEIGHTS
    )

    model_img = calibrate_on_edge(
        model_img,
        calibration_df,
        VAL_TRANSFORM
    )

    (
        img_results,
        img_metrics
    ) = evaluate_model(
        model_img,
        holdout_df,
        VAL_TRANSFORM,
        "InceptionV3 ImageNet Calibrated"
    )

    save_evaluation_results(
        img_results,
        img_metrics,
        "ImageNet_Calibrated"
    )

    img_results = evaluate_against_expert2(
        img_results,
        EXPERT_2_DIR,
        "InceptionV3 ImageNet Calibrated"
    )

    img_results.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "ImageNet_Calibrated",
            "Holdout_Predictions_With_ExpertII.csv"
        ),
        index=False
    )

    img_xai = generate_gradcam_analysis(
        model_img,
        img_results,
        "ImageNet_Calibrated",
        VAL_TRANSFORM
    )

    summarize_xai(
        img_xai,
        "ImageNet_Calibrated"
    )

    # --------------------------------------------------------
    # Run RadImageNet
    # --------------------------------------------------------

    print(
        "\n\n########################################"
    )

    print(
        "RADIMAGENET INITIALIZATION"
    )

    print(
        "########################################"
    )

    model_rad = get_model()

    model_rad = load_weights(
        model_rad,
        RADIMAGENET_WEIGHTS
    )

    model_rad = calibrate_on_edge(
        model_rad,
        calibration_df,
        VAL_TRANSFORM
    )

    (
        rad_results,
        rad_metrics
    ) = evaluate_model(
        model_rad,
        holdout_df,
        VAL_TRANSFORM,
        "InceptionV3 RadImageNet Calibrated"
    )

    save_evaluation_results(
        rad_results,
        rad_metrics,
        "RadImageNet_Calibrated"
    )

    rad_results = evaluate_against_expert2(
        rad_results,
        EXPERT_2_DIR,
        "InceptionV3 RadImageNet Calibrated"
    )

    rad_results.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "RadImageNet_Calibrated",
            "Holdout_Predictions_With_ExpertII.csv"
        ),
        index=False
    )

    rad_xai = generate_gradcam_analysis(
        model_rad,
        rad_results,
        "RadImageNet_Calibrated",
        VAL_TRANSFORM
    )

    summarize_xai(
        rad_xai,
        "RadImageNet_Calibrated"
    )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    comparison = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Cohen's Kappa",
            "MCC",
            "Macro AUC"
        ],

        "ImageNet": [
            img_metrics["accuracy"],
            img_metrics["kappa"],
            img_metrics["mcc"],
            img_metrics["macro_auc"]
        ],

        "RadImageNet": [
            rad_metrics["accuracy"],
            rad_metrics["kappa"],
            rad_metrics["mcc"],
            rad_metrics["macro_auc"]
        ]
    })

    comparison_path = os.path.join(
        OUTPUT_ROOT,
        "ImageNet_vs_RadImageNet_Comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print(
        "\n=========================================="
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=========================================="
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "\n=========================================="
    )

    print(
        "PIPELINE COMPLETE"
    )

    print(
        "=========================================="
    )


# ============================================================
# 25. EXECUTE
# ============================================================

if __name__ == "__main__":

    main()