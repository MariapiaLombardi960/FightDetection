import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor, AutoConfig
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, average_precision_score,precision_score, recall_score, f1_score
import pandas as pd
import h5py
from PIL import Image
import io

# AFTER SAVING THE DATASET IN HDF5 FORMAT

# Select computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Model and processor initialization
# --------------------------------------------------

model_path = "/user/mlombardi/model"

# Load model configuration and adapt it for binary classification
config = AutoConfig.from_pretrained(model_path)
config.num_labels = 2
config.label2id = {"non_fight": 0, "fight": 1}
config.id2label = {0: "non_fight", 1: "fight"}

# Load video processor
processor = VJEPA2VideoProcessor.from_pretrained(model_path)

# Load pretrained V-JEPA2 model and adapt the classification head
model = VJEPA2ForVideoClassification.from_pretrained(
    model_path,
    config=config,
    ignore_mismatched_sizes=True,  # Required to adapt the final classification head
).to(device)

# Number of frames per clip
clip_length = 64

# --------------------------------------------------
# Dataset definition
# --------------------------------------------------

# Dataset class used to load 64-frame clips stored in HDF5 format
class HDF5ClipDataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.file = h5py.File(hdf5_path, "r")
        self.keys = list(self.file.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        grp = self.file[self.keys[index]]

        label = grp.attrs["label"]
        video_id = grp.attrs["video_id"]

        frames = []

        for i in range(clip_length):
            jpeg_bytes = grp[f"frame_{i}"][()]

            # Decode previously stored JPEG frames using PIL
            img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            img_arr = np.array(img)
            frame = torch.from_numpy(img_arr).permute(2, 0, 1).contiguous()

            frames.append(frame)

            # --- Ensure consistent spatial resolution across frames ---
            min_h = min(frame.shape[1] for frame in frames)
            min_w = min(frame.shape[2] for frame in frames)
            frames = [f[:, :min_h, :min_w] for f in frames]

        # Stack frames into a tensor of shape [T, 3, H, W]
        clip_tensor = torch.stack(frames) 

        return clip_tensor, label, video_id

print("Creating training dataset")
train_dataset = HDF5ClipDataset("/user/mlombardi/ClipDatasetTrain.h5")
print("Creating validation dataset")
val_dataset   = HDF5ClipDataset("/user/mlombardi/ClipDatasetVal.h5")

# --------------------------------------------------
# Custom collate function
# --------------------------------------------------

def collate_fn(batch):
    clips, labels, video_ids = zip(*batch)
    return list(clips), torch.tensor(labels), list(video_ids)

# --------------------------------------------------
# DataLoader initialization
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

# --------------------------------------------------
# Model fine-tuning strategy
# --------------------------------------------------

# Freeze all model parameters by default
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last self-attention layer of the pooler
for param in model.pooler.self_attention_layers[-1].parameters():
    param.requires_grad = True

# Unfreeze the final classification head
for param in model.classifier.parameters():
    param.requires_grad = True

# --------------------------------------------------
# Optimizer configuration
# --------------------------------------------------

# A lower learning rate is used for the self-attention layer to avoid catastrophic forgetting,
# while a higher learning rate is assigned to the newly initialized classification head.
optimizer = torch.optim.Adam([
    {"params": model.pooler.self_attention_layers[-1].parameters(), "lr": 1e-5}, 
    {"params": model.classifier.parameters(), "lr": 2e-3} 
])


# --------------------------------------------------
# Threshold optimization
# --------------------------------------------------

# This function searches for the probability threshold that maximizes the F1-score
def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 21) 
    best_t, best_f1 = 0.5, 0
    for t in thresholds:
        preds = (y_prob > t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

# --------------------------------------------------
# Model evaluation
# --------------------------------------------------

# This function computes the evaluation metrics and returns a dictionary containing
# ROC-AUC, Average Precision, Precision, Recall, F1-score, and their abnormal-only variants.
def evaluate_metrics(model, loader, processor, device):
    model.eval()
    all_labels = []
    all_probs = []
    all_video_ids = []

    with torch.no_grad():  
        for batch in tqdm(loader, desc="validation", ncols=100):
            if len(batch) == 3:
                clips, labels, video_ids = batch
            else:
                raise ValueError("Il DataLoader deve restituire clip_frames, clip_label, video_id")
            
            processed = processor(clips, return_tensors="pt")
            processed = {k: v.to(device) for k, v in processed.items()}
            labels = labels.to(device)

            # Forward pass
            with autocast("cuda"):
                outputs = model(**processed)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]  # 'fight' class probability

            # Detach tensors before converting to NumPy
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_video_ids.extend(video_ids)

    # Convert lists to NumPy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_video_ids = np.array(all_video_ids)

    unique_values, counts = np.unique(all_labels, return_counts=True)
    print(f"Labels present in the validation set:{unique_values}, with occurrences {counts}")

    # Determine the optimal classification threshold
    best_threshold, best_f1 = find_best_threshold(all_labels, all_probs)
    print(f"Best threshold = {best_threshold:.3f}, F1-score = {best_f1:.3f}")

    # Binarize predictions using the optimal threshold
    preds_bin = (all_probs > best_threshold).astype(int)

    # Global metrics
    metrics = {}
    metrics['ROC-AUC'] = roc_auc_score(all_labels, all_probs)
    metrics['AP'] = average_precision_score(all_labels, all_probs)
    metrics['Precision'] = precision_score(all_labels, preds_bin)
    metrics['Recall'] = recall_score(all_labels, preds_bin)
    metrics['F1'] = f1_score(all_labels, preds_bin)

    # Abnormal-only metrics
    df = pd.DataFrame({
        "video_id": all_video_ids,
        "label": all_labels,
        "pred": all_probs
    })

    # Select videos containing at least one positive clip
    abnormal_videos = df.groupby("video_id")["label"].max()
    abnormal_videos = abnormal_videos[abnormal_videos == 1].index

    apa_list = []
    auca_list = []

    for vid in abnormal_videos:
        df_v = df[df["video_id"] == vid]

        # Safety check
        if df_v["label"].sum() == 0:
            continue

        # Average Precision for the single video
        ap_v = average_precision_score(df_v["label"], df_v["pred"])
        apa_list.append(ap_v)

        # ROC-AUC requires at least one positive and one negative sample
        if df_v["label"].nunique() == 2:
            auc_v = roc_auc_score(df_v["label"], df_v["pred"])
            auca_list.append(auc_v)

    # Mean values over all abnormal videos
    metrics["APA"] = np.mean(apa_list) if len(apa_list) > 0 else np.nan
    metrics["AUCA"] = np.mean(auca_list) if len(auca_list) > 0 else np.nan


    return metrics

# --------------------------------------------------
# Checkpoint configuration
# --------------------------------------------------
checkpoint_dir = "/user/mlombardi/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
last_checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")

# Gradient scaler for mixed-precision training
scaler = GradScaler("cuda")

best_apa = 0.0
start_epoch = 1
num_epochs = 50

# Gradient accumulation steps:
# effective batch size = batch_size * accum_steps = 4 * 4 = 16.
# This strategy is adopted due to GPU memory constraints.
accum_steps = 4  

# Early stopping configuration
patience = 5  # number of consecutive epochs without improvement
epochs_without_improvement = 0

# --------------------------------------------------
# Resume training from checkpoint (if available)
# --------------------------------------------------

if os.path.exists(last_checkpoint_path):
    checkpoint = torch.load(last_checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_apa = checkpoint.get("best_apa", 0.0)

    print(f"Resuming training from epoch {start_epoch-1}, best APA = {best_apa:.4f}")

# --------------------------------------------------
# Training loop
# --------------------------------------------------

print("STARTING TRAINING")

for epoch in range(start_epoch, num_epochs + 1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for i, (clips, labels, video_ids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        processed = processor(clips, return_tensors="pt")
        processed = {k: v.to(device) for k, v in processed.items()}
        labels = labels.to(device)
        
        # Forward pass with mixed precision
        with autocast("cuda"):
            outputs = model(**processed, labels=labels)
            loss = outputs.loss / accum_steps  # scale loss for gradient accumulation

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        running_loss += loss.item()

        # Perform optimizer step every accum_steps iterations
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            running_loss = 0.0

    # Handle the case where the last batch is not a multiple of accum_steps
    if (i + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        running_loss = 0.0
    
    # --------------------------------------------------
    # Validation phase
    # --------------------------------------------------
    metrics = evaluate_metrics(model, val_loader, processor, device)
    print(metrics)
    current_apa = metrics["APA"]

    # --------------------------------------------------
    # Best model selection & early stopping logic
    # --------------------------------------------------

    if current_apa > best_apa:
        best_apa = current_apa
        epochs_without_improvement = 0  # reset counter

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_apa": best_apa
        }, best_model_path)

        print(f" New BEST MODEL saved (APA = {current_apa:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"No improvement in APA for {epochs_without_improvement} consecutive epoch(s)")

    # --------------------------------------------------
    # Save last checkpoint
    # --------------------------------------------------
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_apa": best_apa
    }, last_checkpoint_path)

    print(" Last checkpoint updated.")

    # --------------------------------------------------
    # Early stopping condition
    # --------------------------------------------------

    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {patience} epochs without APA improvement.")
        break
