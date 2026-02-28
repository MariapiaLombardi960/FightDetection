import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor, AutoConfig
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import pandas as pd
import h5py
from PIL import Image
import io
import time

# Select computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 1) Load video processor and fine-tuned model
# --------------------------------------------------

# Path to the pretrained V-JEPA2 model
model_path = "/user/mlombardi/model"  

# Load model configuration and adapt it to binary classification
config = AutoConfig.from_pretrained(model_path)
config.num_labels = 2
config.label2id = {"non_fight": 0, "fight": 1}
config.id2label = {0: "non_fight", 1: "fight"}

# Load the video processor associated with V-JEPA2
processor = VJEPA2VideoProcessor.from_pretrained(model_path)

# Load the model and adapt the classification head
model = VJEPA2ForVideoClassification.from_pretrained(
    model_path,
    config=config,
    ignore_mismatched_sizes=True,  # Required to adapt the final classification head
).to(device)

# Load the best checkpoint obtained during training
checkpoint_dir = "/user/mlombardi/checkpoints"
best_model_path = os.path.join(checkpoint_dir, "best_model.pt")


# Restore model weights fine-tuned for the fight detection task
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# --------------------------------------------------
# 2) Dataset definition for test evaluation
# --------------------------------------------------

# Number of frames per clip (as required by V-JEPA2)
clip_length = 64

class HDF5ClipDataset(Dataset):
    """
    Dataset class for loading video clips stored in HDF5 format.
    Each sample corresponds to a clip of fixed length (64 frames),
    stored as JPEG-encoded images.
    """
    def __init__(self, hdf5_path):
        self.file = h5py.File(hdf5_path, "r")
        self.keys = list(self.file.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        grp = self.file[self.keys[index]]

        # Retrieve clip-level label and video identifier
        label = grp.attrs["label"]
        video_id = grp.attrs["video_id"]

        frames = []
        for i in range(clip_length):
            jpeg_bytes = grp[f"frame_{i}"][()]
            # Decode JPEG-encoded frames using PIL
            img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            arr = np.array(img)
            frame = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            frames.append(frame)

        # Ensure spatial consistency across frames
        min_h = min(f.shape[1] for f in frames)
        min_w = min(f.shape[2] for f in frames)
        frames = [f[:, :min_h, :min_w] for f in frames]

        # Stack frames into a clip tensor of shape [T, 3, H, W]
        clip_tensor = torch.stack(frames)  
        return clip_tensor, label, video_id

# Instantiate the test dataset
test_dataset = HDF5ClipDataset("/user/mlombardi/ClipDatasetTestReview.h5")

def collate_fn(batch):
    clips, labels, video_ids = zip(*batch)
    return list(clips), torch.tensor(labels), list(video_ids)

# DataLoader for test-time inference (batch size = 1)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# --------------------------------------------------
# 3) Evaluation metrics
# --------------------------------------------------

def find_best_threshold(y_true, y_prob):
    """
    Finds the probability threshold that maximizes the F1-score.
    """
    thresholds = np.linspace(0.1, 0.9, 21)
    best_t, best_f1 = 0.5, 0
    for t in thresholds:
        preds = (y_prob > t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def evaluate_metrics(model, loader, processor, device):
    """
    Performs inference on the test set and computes both
    classification and real-time performance metrics.
    """
    model.eval()

    all_labels = []
    all_probs = []
    all_video_ids = []

    # Latency measurements (in milliseconds)
    preprocess_latencies = []
    model_latencies = []
    clip_latencies = []  # total latency per clip (preprocessing + inference)

    with torch.no_grad():
        for clips, labels, video_ids in tqdm(loader, desc="Inferenza", ncols=100):

            t0 = time.perf_counter()

            processed = processor(clips, return_tensors="pt")
            processed = {k: v.to(device) for k, v in processed.items()}
            labels = labels.to(device)

            t1 = time.perf_counter()
            preprocess_latency = (t1 - t0) * 1000
            preprocess_latencies.append(preprocess_latency / len(clips)) 

            # Synchronize GPU before inference for accurate timing
            if device.type == "cuda":
                torch.cuda.synchronize()

            t2 = time.perf_counter()

            with autocast("cuda"):
                outputs = model(**processed)

            if device.type == "cuda":
                torch.cuda.synchronize()
            
            # Extract probability of the positive class ("fight")
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]

            t3 = time.perf_counter()

            model_latency = (t3 - t2) * 1000  
            model_latencies.append(model_latency / len(clips)) 

            # Total latency per clip
            total_latency = (t3 - t0) * 1000
            batch_size_current = len(clips) 
            latency_per_clip = total_latency / batch_size_current
            clip_latencies.append(latency_per_clip)

            # Store predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_video_ids.extend(video_ids)

    # --------------------------------------------------
    # Standard classification metrics
    # --------------------------------------------------
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_video_ids = np.array(all_video_ids)

    best_thr, best_f1 = find_best_threshold(all_labels, all_probs)
    print(f"Best threshold = {best_thr:.3f}, F1-score = {best_f1:.3f}")

    preds_bin = (all_probs > best_thr).astype(int)

    metrics = {
        "ROC-AUC": roc_auc_score(all_labels, all_probs),
        "AP": average_precision_score(all_labels, all_probs),
        "Precision": precision_score(all_labels, preds_bin),
        "Recall": recall_score(all_labels, preds_bin),
        "F1": f1_score(all_labels, preds_bin),
    }

    # --------------------------------------------------
    # Abnormal-video metrics (APA and AUCA)
    # --------------------------------------------------
    df = pd.DataFrame({
        "video_id": all_video_ids,
        "label": all_labels,
        "pred": all_probs
    })

    abnormal_vids = df.groupby("video_id")["label"].max()
    abnormal_vids = abnormal_vids[abnormal_vids == 1].index
    apa_list = []
    auca_list = []

    for vid in abnormal_vids:
        df_v = df[df["video_id"] == vid]

        # Safety check: skip videos with no positive clips
        if df_v["label"].sum() == 0:
            continue

        # Average Precision for a single abnormal video
        ap_v = average_precision_score(df_v["label"], df_v["pred"])
        apa_list.append(ap_v)

        # ROC-AUC for a single abnormal video
        if df_v["label"].nunique() == 2:
            auc_v = roc_auc_score(df_v["label"], df_v["pred"])
            auca_list.append(auc_v)

    metrics["APA"] = np.mean(apa_list) if len(apa_list) > 0 else np.nan
    metrics["AUCA"] = np.mean(auca_list) if len(auca_list) > 0 else np.nan

    # --------------------------------------------------
    # Real-time performance metrics
    # --------------------------------------------------
    clip_latencies = np.array(clip_latencies)
    preprocess_latencies = np.array(preprocess_latencies)
    model_latencies = np.array(model_latencies)

    metrics["Preprocess_mean_ms"] = preprocess_latencies.mean() 
    metrics["Model_mean_ms"] = model_latencies.mean() 
    metrics["Latency_mean_ms"] = clip_latencies.mean()  
    
    metrics["Latency_std_ms"] = clip_latencies.std() 
    metrics["Latency_p95_ms"] = np.percentile(clip_latencies, 95)  
    
    # Throughput metrics
    metrics["Clips_per_second"] = 1000.0 / metrics["Latency_mean_ms"]  
    metrics["Frames_per_second"] = clip_length * metrics["Clips_per_second"] 

    ORIGINAL_FPS = 5  # original video frame rate
    metrics["RTPR"] = metrics["Frames_per_second"] / ORIGINAL_FPS 
    # Real-Time Processing Ratio (RTPR) > 1 indicates real-time capability

    return metrics


# --------------------------------------------------
# 4) Final test evaluation
# --------------------------------------------------
metrics = evaluate_metrics(model, test_loader, processor, device)
print("\n=== FINAL TEST METRICS ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

