import os
import re
import numpy as np
import torch
from decord import VideoReader, cpu
import h5py
from tqdm import tqdm
from transformers import AutoVideoProcessor, AutoModel

# =========================
# CONFIGURATION
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor from local folder
model_path = "/user/mlombardi/model"
model = AutoModel.from_pretrained(model_path).to(device)
processor = AutoVideoProcessor.from_pretrained(model_path)

model.half()  # Use FP16 for faster inference
model.eval()  # Set model to evaluation mode

clip_length = 64
target_fps = 5
label_map = {"A":0, "B1":1, "B2":2, "B4":3, "B5":4, "B6":5, "G":6}

videos_dir = "/user/mlombardi/Videos/test_set_review"
annotations_path = "/user/mlombardi/test_annotations_review.txt"

# =========================
# BATCH PARAMETERS
# =========================
batch_size = 50   # Number of videos per batch
batch_index = 15  # Which batch to process in this run

# =========================
# HELPER FUNCTIONS
# =========================
def parse_video_label(video_id):
    """Extract labels from video_id string."""
    label_str = video_id.split("_label_")[1]
    labels = label_str.split("-")
    return [l for l in labels if l != '0']

def parse_annotation_line(line):
    """
    Parse a line from the annotation file.
    Returns video_id and list of intervals (start, end, type)
    """
    parts = [p for p in line.strip().split() if p]
    video_id = parts[0]
    intervals = []
    i = 1

    while i + 1 < len(parts):
        start_matches = re.findall(r"\d+", parts[i])
        end_matches = re.findall(r"\d+", parts[i + 1])
        if not start_matches or not end_matches:
            i += 1
            continue
        start = int(start_matches[0])
        end = int(end_matches[0])

        a_type = None
        if i + 2 < len(parts) and re.match(r"\(.+\)", parts[i + 2]):
            a_type = parts[i + 2].strip("()")
            i += 1

        intervals.append((start, end, a_type))
        i += 2

    return video_id, intervals

def create_frame_labels(num_frames, intervals, default_type, label_map):
    """
    Create frame-level labels based on annotated intervals.
    Assigns a label to each frame in the video.
    """
    frame_labels = np.zeros(num_frames, dtype=int)
    for (start, end, a_type) in intervals:
        types = [a_type] if a_type is not None else [default_type]
        chosen = "B1" if "B1" in types else types[0]
        frame_labels[start:end+1] = label_map[chosen]
    return frame_labels

def pad_clip(indices, target_len=64):
    """Pad clip indices to target length by repeating the last frame if needed."""
    if len(indices) < target_len:
        pad_len = target_len - len(indices)
        indices = np.concatenate([indices, np.full(pad_len, indices[-1])])
    return indices

# =========================
# PREPARE VIDEO FILES FOR BATCH
# =========================
video_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]
video_files.sort()  # Important for consistent batching

num_batches = len(video_files) // batch_size + (len(video_files) % batch_size > 0)
print(f"Total batches: {num_batches}")
start_idx = batch_index * batch_size
end_idx = min(start_idx + batch_size, len(video_files))
subset_files = video_files[start_idx:end_idx]

print(f"Batch {batch_index+1}/{num_batches}: video {start_idx} → {end_idx} (tot {len(subset_files)})")

# =========================
# OUTPUT HDF5 FILE
# =========================
output_h5 = "/user/mlombardi/clips_features_testset_review.h5"
f_out = h5py.File(output_h5, "a")

# Resume from last saved clip
clip_counter = len(f_out.keys())
print(f"Resuming from clip_counter = {clip_counter}")

existing_videos = {f_out[k].attrs["video_id"] for k in f_out.keys()}
print(f"Videos already in file: {len(existing_videos)}")

# =========================
# LOAD ANNOTATIONS
# =========================
with open(annotations_path, "r") as f:
    lines = f.readlines()

# =========================
# FEATURE EXTRACTION LOOP
# =========================

for video_file in tqdm(subset_files, desc="Video loop"):
    video_id = os.path.splitext(video_file)[0]

    if video_id in existing_videos:
        print(f"Video {video_id} already processed, skipping.")
        continue

    labels = parse_video_label(video_id)
    default_type = labels[0] if labels else "A"

    video_path = os.path.join(videos_dir, video_file)
    print(f"Processing: {video_path}")
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        continue

    # Determine whether to skip annotation search
    if "A" in labels:
        intervals = []
        skip_search = True
    else:
        skip_search = False
        found = False
        for line in lines:
            video_id_raw, intervals_tmp = parse_annotation_line(line)
            if video_id_raw == video_id:  
                intervals = intervals_tmp
                found = True
                break
        if not found:
            print(f"No annotation found for {video_id}, skipping.")
            continue

    # Load video
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    fps = vr.get_avg_fps()

    # Frame-level labels
    frame_labels = np.zeros(num_frames, dtype=int) if skip_search else create_frame_labels(num_frames, intervals, default_type, label_map)
    
    # Sample frames according to target FPS
    step = max(1, int(fps / target_fps))
    frame_indices = np.arange(0, num_frames, step)

    # Process video in clips
    for i in range(0, len(frame_indices), clip_length):
        clip_idx = frame_indices[i:i+clip_length]
        clip_idx = pad_clip(clip_idx, clip_length)

        # Load clip frames and convert to tensor
        clip_frames = torch.stack([
        torch.tensor(vr[idx].asnumpy()).permute(2, 0, 1).to(torch.uint8)
        for idx in clip_idx
        ])

        # Process video frames for model input
        video_tensor = processor(clip_frames, return_tensors="pt").to(device)

        # Convert to FP16
        video_tensor = {k: v.half() for k, v in video_tensor.items()}

        with torch.no_grad():
            feats = model.get_vision_features(**video_tensor)  # [1, 18432, 1408]

        # ===== MODIFICATION: reduce tokens → tubelets =====
        feats_per_tubelet = feats.view(1, 32, 576, 1408).mean(dim=2)  # [1, 32, 1408]

        # Clip-level label: dominant / fight if ≥15%
        clip_labels = frame_labels[clip_idx]
        unique, counts = np.unique(clip_labels, return_counts=True)
        dominant_label = unique[np.argmax(counts)]
        rissa_ratio = np.mean(clip_labels == label_map["B1"])
        clip_label = 1 if dominant_label == label_map["B1"] or rissa_ratio >= 0.15 else 0

        # Save clip features and metadata in HDF5
        grp = f_out.create_group(f"clip_{clip_counter}")
        grp.create_dataset("features", data=feats_per_tubelet[0].cpu().numpy(), dtype="float16")  # [32, 1408]
        grp.create_dataset("label", data=clip_label, dtype="int")
        grp.attrs["video_id"] = video_id
        grp.attrs["start_frame"] = int(clip_idx[0])
        grp.attrs["end_frame"] = int(clip_idx[-1])

        clip_counter += 1
        f_out.flush()  # Save to disk immediately

f_out.close()
print(f"\nBatch {batch_index} completed! Total clips in file: {clip_counter}")
