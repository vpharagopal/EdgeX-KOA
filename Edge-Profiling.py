# ============================================================
# EdgeX-KOA: Standardized Deployment & Complexity Profiling
# ============================================================
# Protocol:
# 1. Theoretical Complexity: All 6 Models (Params, GMACs, GFLOPs, MB)
# 2. InceptionV3 Stage-Wise Latency: CPU vs. GPU
# 3. Edge-Model CPU Comparison: InceptionV3, MobileNetV2, MobileNetV3, EfficientNet-B0
# 4. Dynamic RAM/VRAM Profiling with Reproducible Selection
# ============================================================

import os
import time
import copy
import warnings
import numpy as np
import pandas as pd
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from skimage.exposure import match_histograms
import cv2

warnings.filterwarnings("ignore")

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("WARNING: 'thop' is not installed. MAC/FLOP profiling will be skipped.")


# ============================================================
# 1. CONFIGURATION & PATHS
# ============================================================

NUM_CLASSES = 5

TEST_DIR = r"C:\Users\vphar\KOA-Chen-Kaggle\archive\test"
OAI_REFERENCE_PATH = r"C:\Users\vphar\KOA-Chen-Kaggle\archive\train\0\9001695L.png"

OUTPUT_DIR = (
    r"C:\Users\vphar\KOA-Chen-Kaggle"
    r"\PhD Objectives\Re-Execution-Without-BalancedSet-New"
    r"\EdgeX-KOA-Profiling"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Actual Checkpoint Mapping
CHECKPOINTS = {
    "DenseNet-121": r"C:\Users\vphar\KOA-Chen-Kaggle\PhD Objectives\Re-Execution-Without-BalancedSet-New\ImageNet\DenseNet121-Img-Files\radx_koa_densenet_rad_model.pth",
    "ResNet-50": r"C:\Users\vphar\KOA-Chen-Kaggle\PhD Objectives\Re-Execution-Without-BalancedSet-New\ImageNet\ResNet50-Img-Files\radx_koa_resnet_rad_model.pth",
    "InceptionV3": r"C:\Users\vphar\KOA-Chen-Kaggle\PhD Objectives\Re-Execution-Without-BalancedSet-New\ImageNet\InceptionV3-Img-Files\radx_koa_inceptionv3_rad_model.pth",
    "MobileNetV2": r"C:\Users\vphar\KOA-Chen-Kaggle\PhD Objectives\Re-Execution-Without-BalancedSet-New\Edge Models\MobileNetV2-Img-Files\radx_koa_MobileNetV2_model.pth",
    "MobileNetV3": r"C:\Users\vphar\KOA-Chen-Kaggle\PhD Objectives\Re-Execution-Without-BalancedSet-New\Edge Models\MobileNetV3-Img-Files\radx_koa_MobileNetV3_model.pth",
    "EfficientNet-B0": r"C:\Users\vphar\KOA-Chen-Kaggle\PhD Objectives\Re-Execution-Without-BalancedSet-New\Edge Models\EfficientNetB0-Img-Files\radx_koa_EfficientNetB0_model.pth",
}

MODEL_RESOLUTIONS = {
    "DenseNet-121": 224,
    "ResNet-50": 224,
    "InceptionV3": 299,
    "MobileNetV2": 224,
    "MobileNetV3": 224,
    "EfficientNet-B0": 224
}

CPU_DEVICE = torch.device("cpu")
GPU_AVAILABLE = torch.cuda.is_available()
GPU_DEVICE = torch.device("cuda") if GPU_AVAILABLE else None


# ============================================================
# 2. EXACT ARCHITECTURE DEFINITIONS
# ============================================================

class CustomDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.densenet121(weights=None)
        self.base_model = base
        num_features = base.classifier.in_features
        self.fc1 = nn.Linear(num_features, 1024)
        self.dropout = nn.Dropout(p=0.57)
        self.fc2 = nn.Linear(1024, NUM_CLASSES)

    def forward(self, x):
        x = self.base_model.features(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CustomResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=None)
        self.base_model = nn.Sequential(*list(base.children())[:-1])
        num_features = base.fc.in_features
        self.fc1 = nn.Linear(num_features, 1024)
        self.dropout = nn.Dropout(p=0.63)
        self.fc2 = nn.Linear(1024, NUM_CLASSES)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        return self.fc2(x)


class CustomInceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.inception_v3(weights=None, init_weights=False, aux_logits=False)
        base.AuxLogits = None
        self.base_model = base
        num_features = base.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.28),
            nn.Linear(1024, NUM_CLASSES)
        )

    def forward(self, x):
        outputs = self.base_model(x)
        return outputs.logits if hasattr(outputs, "logits") else outputs


class CustomMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1280, 1024)
        self.dropout = nn.Dropout(p=0.63)
        self.fc2 = nn.Linear(1024, NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        return self.fc2(x)


class CustomMobileNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v3_large(weights=None)
        self.model.classifier[2] = nn.Dropout(p=0.63, inplace=True)
        self.model.classifier[3] = nn.Linear(1280, NUM_CLASSES)

    def forward(self, x):
        return self.model(x)


class CustomEfficientNetB0(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights=None)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, NUM_CLASSES)

    def forward(self, x):
        return self.model(x)


def create_model(model_name):
    factory = {
        "DenseNet-121": CustomDenseNet,
        "ResNet-50": CustomResNet,
        "InceptionV3": CustomInceptionV3,
        "MobileNetV2": CustomMobileNetV2,
        "MobileNetV3": CustomMobileNetV3,
        "EfficientNet-B0": CustomEfficientNetB0,
    }
    if model_name not in factory:
        raise ValueError(f"Unknown model: {model_name}")
    return factory[model_name]()


# ============================================================
# 3. PREPROCESSING & DATA UTILITIES
# ============================================================

class ApplyCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return Image.fromarray(clahe.apply(img_np)).convert("RGB")


class HistogramMatcher:
    def __init__(self, reference_path):
        ref_img = Image.open(reference_path).convert("L")
        self.reference_arr = np.array(ref_img)

    def __call__(self, img):
        img_arr = np.array(img.convert("L"))
        matched_arr = match_histograms(img_arr, self.reference_arr)
        matched_arr = np.clip(matched_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(matched_arr).convert("RGB")


histogram_matcher = HistogramMatcher(OAI_REFERENCE_PATH)
clahe_transformer = ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def preprocess_image(image_path, input_size):
    timings = {}

    t0 = time.perf_counter()
    img = Image.open(image_path).convert("RGB")
    timings["Loading"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    img = histogram_matcher(img)
    timings["Histogram Matching"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    img = clahe_transformer(img)
    timings["CLAHE"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    img = img.resize((input_size, input_size), Image.Resampling.BILINEAR)
    timings["Resize"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    tensor = transforms.ToTensor()(img)
    tensor = normalize(tensor).unsqueeze(0)
    timings["Tensor + Normalize"] = (time.perf_counter() - t0) * 1000

    return tensor, timings

# FIX 1: REPRODUCIBLE RANDOM IMAGE SELECTION
def get_test_images(test_dir, max_images=300, seed=42):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    paths = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                paths.append(os.path.join(root, file))
    paths.sort() # Ensure stable basis before random selection
    
    if len(paths) == 0:
        raise RuntimeError(f"No images found in:\n{test_dir}")
        
    rng = np.random.default_rng(seed)
    selected_paths = rng.choice(paths, size=min(max_images, len(paths)), replace=False).tolist()
    return selected_paths


def load_model_checkpoint(model, path):
    if not os.path.exists(path):
        print(f"Warning: Checkpoint not found: {path}. Profiling uninitialized weights.")
        return model
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    return model


# ============================================================
# 4. LEVEL 1: THEORETICAL COMPLEXITY PROFILING (ALL 6 MODELS)
# ============================================================

def run_complexity_profiling():
    print("\n" + "=" * 70)
    print("LEVEL 1: SIX-MODEL THEORETICAL COMPLEXITY PROFILING")
    print("=" * 70)

    results = []
    for model_name, input_size in MODEL_RESOLUTIONS.items():
        model = create_model(model_name)
        ckpt_path = CHECKPOINTS[model_name]
        model = load_model_checkpoint(model, ckpt_path)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        state_mb = sum(t.numel() * t.element_size() for t in model.state_dict().values() if torch.is_tensor(t)) / (1024 ** 2)
        ckpt_mb = os.path.getsize(ckpt_path) / (1024 ** 2) if os.path.exists(ckpt_path) else np.nan

        macs, flops = np.nan, np.nan
        if THOP_AVAILABLE:
            dummy = torch.randn(1, 3, input_size, input_size)
            macs, _ = profile(copy.deepcopy(model).cpu(), inputs=(dummy,), verbose=False)
            flops = 2 * macs

        results.append({
            "Model": model_name,
            "Input Resolution": f"{input_size}x{input_size}",
            "Parameters (M)": round(total_params / 1e6, 3),
            "Trainable Params (M)": round(trainable_params / 1e6, 3),
            # FIX 2: PRECISE TERMINOLOGY FOR STATIC FOOTPRINT
            "StateDict Memory (MB)": round(state_mb, 2),
            "Checkpoint (MB)": round(ckpt_mb, 2) if not np.isnan(ckpt_mb) else np.nan,
            "GMACs": round(macs / 1e9, 3) if not np.isnan(macs) else np.nan,
            "GFLOPs": round(flops / 1e9, 3) if not np.isnan(flops) else np.nan,
        })
        del model

    df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "EdgeX_KOA_Six_Model_Complexity.csv")
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved complexity metrics to:\n{out_path}")
    return df


# ============================================================
# 5. LEVEL 2: LATENCY & DYNAMIC RAM PROFILING ENGINE
# ============================================================

def benchmark_model_execution(model_name, device, image_paths, warmup=30, detailed_breakdown=False):
    dev_label = str(device).upper()
    print(f"\nBenchmarking {model_name} on {dev_label}...")

    input_size = MODEL_RESOLUTIONS[model_name]
    model = create_model(model_name)
    model = load_model_checkpoint(model, CHECKPOINTS[model_name]).to(device)
    model.eval()

    # FIX 3: RESET CUDA MEMORY STATISTICS
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up
    with torch.no_grad():
        for path in image_paths[:warmup]:
            tensor, _ = preprocess_image(path, input_size)
            tensor = tensor.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            _ = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Profiling Loop
    records = {k: [] for k in ["Loading", "Histogram Matching", "CLAHE", "Resize", "Tensor + Normalize", "Host-to-Device", "Model Inference", "Softmax", "End-to-End"]}
    peak_ram_mb = []

    process = psutil.Process(os.getpid())

    for path in image_paths[warmup:]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_start = time.perf_counter()

        # 1. Preprocessing Stages
        tensor, prep_times = preprocess_image(path, input_size)
        for k, v in prep_times.items():
            records[k].append(v)

        # 2. Host-to-Device Transfer
        t0 = time.perf_counter()
        tensor = tensor.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        records["Host-to-Device"].append((time.perf_counter() - t0) * 1000)

        # 3. Model Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        records["Model Inference"].append((time.perf_counter() - t0) * 1000)

        # 4. Softmax
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = torch.softmax(logits, dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        records["Softmax"].append((time.perf_counter() - t0) * 1000)

        # 5. End-to-End Total
        records["End-to-End"].append((time.perf_counter() - total_start) * 1000)

        # 6. Dynamic Memory Sampling
        if device.type == "cuda":
            peak_ram_mb.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
        else:
            peak_ram_mb.append(process.memory_info().rss / (1024 ** 2))

    del model
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()

    # FIX 4: PRECISE TERMINOLOGY FOR RUNTIME MEMORY
    memory_col_name = "Peak VRAM (MB)" if device.type == "cuda" else "Peak Process RSS (MB)"

    if detailed_breakdown:
        summary = []
        for stage, vals in records.items():
            summary.append({
                "Device": dev_label,
                "Stage": stage,
                "Mean (ms)": round(np.mean(vals), 2),
                "SD (ms)": round(np.std(vals, ddof=1), 2),
                "Median (ms)": round(np.median(vals), 2),
                "P95 (ms)": round(np.percentile(vals, 95), 2),
            })
        return pd.DataFrame(summary), float(np.max(peak_ram_mb))
    else:
        return {
            "Model": model_name,
            "Device": dev_label,
            "Inference Mean (ms)": round(np.mean(records["Model Inference"]), 2),
            "Inference P95 (ms)": round(np.percentile(records["Model Inference"], 95), 2),
            "End-to-End Mean (ms)": round(np.mean(records["End-to-End"]), 2),
            "End-to-End P95 (ms)": round(np.percentile(records["End-to-End"], 95), 2),
            "Throughput (FPS)": round(1000.0 / np.mean(records["End-to-End"]), 2),
            memory_col_name: round(float(np.max(peak_ram_mb)), 2)
        }


# ============================================================
# 6. MAIN EXECUTION PIPELINE
# ============================================================

if __name__ == "__main__":
    
    test_images = get_test_images(TEST_DIR, max_images=300, seed=42)
    print(f"Randomly selected {len(test_images)} test images (Seed: 42) for profiling.")

    # 1. Theoretical Complexity Table (All 6 Models)
    complexity_df = run_complexity_profiling()

    # 2. Detailed InceptionV3 Breakdown (CPU vs GPU)
    print("\n" + "=" * 70)
    print("LEVEL 2A: INCEPTIONV3 STAGE-WISE BREAKDOWN (CPU VS GPU)")
    print("=" * 70)
    inc_cpu_df, inc_cpu_ram = benchmark_model_execution("InceptionV3", CPU_DEVICE, test_images, detailed_breakdown=True)
    timing_tables = [inc_cpu_df]

    if GPU_AVAILABLE:
        inc_gpu_df, inc_gpu_ram = benchmark_model_execution("InceptionV3", GPU_DEVICE, test_images, detailed_breakdown=True)
        timing_tables.append(inc_gpu_df)

    inc_timing_df = pd.concat(timing_tables, ignore_index=True)
    inc_out = os.path.join(OUTPUT_DIR, "InceptionV3_CPU_GPU_Detailed_Timing.csv")
    inc_timing_df.to_csv(inc_out, index=False)
    print(inc_timing_df.to_string(index=False))

    # 3. Edge-Model CPU Latency Comparison
    print("\n" + "=" * 70)
    print("LEVEL 2B: EDGE-MODEL CPU LATENCY COMPARISON")
    print("=" * 70)
    edge_models = ["InceptionV3", "MobileNetV2", "MobileNetV3", "EfficientNet-B0"]
    edge_summary = []

    for m in edge_models:
        res = benchmark_model_execution(m, CPU_DEVICE, test_images, detailed_breakdown=False)
        edge_summary.append(res)

    edge_df = pd.DataFrame(edge_summary)
    edge_out = os.path.join(OUTPUT_DIR, "Edge_Models_CPU_Latency_Comparison.csv")
    edge_df.to_csv(edge_out, index=False)
    print(edge_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE: All tables saved in:")
    print(OUTPUT_DIR)
    print("=" * 70)