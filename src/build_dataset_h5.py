import os
import re
import numpy as np
import torch
from decord import VideoReader, cpu
from sklearn.model_selection import train_test_split
import h5py
from PIL import Image
import io

# --------------------------------------------------
# Global configuration
# --------------------------------------------------

# Length of the video clip expected by the V-JEPA2 model
clip_length = 64

# Target frame rate used for temporal subsampling
target_fps = 5

# Mapping between anomaly labels and integer class indices
label_map = {"A": 0, "B1": 1, "B2": 2, "B4": 3, "B5": 4, "B6": 5, "G": 6}

# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def load_and_clean_annotations(annotation_path):
    """
    Loads and cleans annotation lines from a text file by
    removing empty lines and Unicode BOM characters.
    """
    with open(annotation_path, "r") as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        if line is None:
            continue
        s = line.lstrip("\ufeff").strip()
        if s:
            cleaned.append(s)

    print(f"Valid annotation lines: {len(cleaned)}")
    return cleaned


def parse_video_labels(video_id):
    """
    Parses anomaly labels encoded in the video filename.
    Labels equal to '0' (i.e., no anomaly) are ignored.
    """
    label_str = video_id.split("_label_")[1]
    labels = label_str.split("-")
    return [l for l in labels if l != "0"]


def parse_annotation_line(line):
    """
    Parses a single annotation line.

    Returns:
        video_id (str)
        intervals (list of tuples): (start_frame, end_frame, anomaly_type)
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


def create_frame_labels(num_frames, intervals, default_type):
    """
    Generates frame-level labels based on annotation intervals.
    In case multiple anomalies are present in the same frame, the fight anomaly (B1)
    is prioritized; otherwise, the first anomaly in the list is assigned.
    """
    frame_labels = np.zeros(num_frames, dtype=int)

    for start, end, a_type in intervals:
        types = [a_type] if a_type else [default_type]
        chosen = "B1" if "B1" in types else types[0]
        frame_labels[start:end + 1] = label_map[chosen]

    return frame_labels


def encode_jpeg(tensor_chw):
    """
    Encodes a CHW tensor into a JPEG byte array for compact storage.
    """
    img = Image.fromarray(tensor_chw.permute(1, 2, 0).numpy())
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# --------------------------------------------------
# Core processing functions
# --------------------------------------------------

def process_video_to_hdf5(video_file, hdf5_file, start_idx, annotation_lines):
    """
    Processes a single video and stores its clips and labels in HDF5 format.
    """
    video_id_clean = os.path.splitext(os.path.basename(video_file))[0]

    # List of anomaly types associated with the current video
    labels = parse_video_labels(video_id_clean)
    default_type = labels[0] if labels else "A"

    # Open the current video file
    vr = VideoReader(video_file, ctx=cpu(0))
    num_frames = len(vr)
    fps = vr.get_avg_fps()

    # Temporal subsampling step to achieve the target frame rate
    step = max(1, int(fps / target_fps))
    frame_indices = np.arange(0, num_frames, step) # Frames indices to select in the original video

    # ---- Frame-level labels generation----
    if "A" in labels:
        # If the video does not contains anomalies, all frames are labeled as normal.
        frame_labels = np.zeros(num_frames, dtype=int)
    else:
        # If anomalies are present, retrieve the corresponding annotation intervals.
        found = False
        for line in annotation_lines:
            vid, intervals_tmp = parse_annotation_line(line)
            if vid == video_id_clean:
                intervals = intervals_tmp
                found = True
                break

        # Generate frame-level labels based on the annotation intervals
        frame_labels = (
            create_frame_labels(num_frames, intervals, default_type)
            if found else np.zeros(num_frames, dtype=int)
        )

    key = start_idx

    # ---- Clip generation ----
    for clip_start in range(0, len(frame_indices), clip_length):
        clip_idx = frame_indices[clip_start:clip_start + clip_length]

        if len(clip_idx) < clip_length:
            # If the last clip is shorter than the expected length,replicate the last frame to reach the required size
            clip_idx = np.pad(
                clip_idx,
                (0, clip_length - len(clip_idx)),
                mode="edge"
            )

        # Encode frames of the current clip
        jpeg_frames = []
        for f in clip_idx:
            frame = vr[int(f)].asnumpy()
            frame_t = torch.tensor(frame).permute(2, 0, 1).byte()
            jpeg_frames.append(encode_jpeg(frame_t))

        # ---- Clip-level label assignment ----
        clip_frame_labels = frame_labels[clip_idx]
        unique, counts = np.unique(clip_frame_labels, return_counts=True)
        dominant = unique[np.argmax(counts)]

        # Ratio of frames labeled as fight anomaly (B1)
        fight_ratio = np.mean(clip_frame_labels == label_map["B1"])

        # Assign clip label:
        # - anomalous if B1 is the dominant label, or
        # - if B1 is present in at least 15% of the clip frames 
        clip_label = (
            1
            if dominant == label_map["B1"] or fight_ratio >= 0.15
            else 0
        )

        # Create a group for each clip and store metadata
        grp = hdf5_file.create_group(str(key))
        grp.attrs["video_id"] = video_id_clean
        grp.attrs["clip_index"] = clip_start // clip_length
        grp.attrs["label"] = clip_label

        # Store frames as JPEG-encoded byte arrays
        for i, jf in enumerate(jpeg_frames):
            grp.create_dataset(f"frame_{i}", data=np.void(jf))

        key += 1

    return key


def build_hdf5(video_list, hdf5_path, annotation_lines):
    """
    Builds an HDF5 dataset by processing a list of videos.
    """
    with h5py.File(hdf5_path, "w") as hdf5_file:
        key = 0
        for i, video_path in enumerate(video_list):
            print(f"[{i+1}/{len(video_list)}] Processing {video_path}")
            key = process_video_to_hdf5(
                video_path, hdf5_file, key, annotation_lines
            )
    print(f"HDF5 dataset created at {hdf5_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # -------- TRAIN / VALIDATION --------
    train_video_dirs = [
        "/user/mlombardi/Videos/1-1004",
        "/user/mlombardi/Videos/1005-2004",
        "/user/mlombardi/Videos/2005-2804",
        "/user/mlombardi/Videos/2805-3319",
        "/user/mlombardi/Videos/3320-3954"
    ]

    train_annotations = load_and_clean_annotations(
        "/user/mlombardi/train_annotations.txt"
    )

    all_videos = []
    # Collect all video files from the specified directories
    for folder in train_video_dirs:
        all_videos += [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".mp4")
        ]

    all_videos = sorted(all_videos)

    video_labels = [
        1 if "B1" in parse_video_labels(
            os.path.splitext(os.path.basename(v))[0]
        ) else 0
        for v in all_videos
    ]

    # Perform a stratified train/validation split to preserve
    # the proportion of anomalous videos in both subsets
    train_videos, val_videos = train_test_split(
        all_videos,
        test_size=0.2,
        random_state=42,
        stratify=video_labels
    )

    build_hdf5(
        train_videos,
        "/user/mlombardi/ClipDatasetTrain.h5",
        train_annotations
    )

    build_hdf5(
        val_videos,
        "/user/mlombardi/ClipDatasetVal.h5",
        train_annotations
    )

    # -------- TEST --------
    test_video_dir = "/user/mlombardi/Videos/test_set_review"

    test_annotations = load_and_clean_annotations(
        "/user/mlombardi/test_annotations_review.txt"
    )

    test_videos = sorted([
        os.path.join(test_video_dir, f)
        for f in os.listdir(test_video_dir)
        if f.endswith(".mp4")
    ])

    build_hdf5(
        test_videos,
        "/user/mlombardi/ClipDatasetTestReview.h5",
        test_annotations
    )

    print("All datasets successfully created")


# Execute main
main()
