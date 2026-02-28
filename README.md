# FightDetection
This project falls within the field of Video Anomaly Detection (VAD). Specifically, it focused on the development of a system for the automatic detection of fight events in video sequences. 
The project involved the fine-tuning of the V-JEPA 2 model, developed and published by Meta in June 2025.
The model used in this work was a version pre-trained on the publicly available Something-Something V2 dataset.
Fine-tuning was performed using the publicly available XD-Violence dataset, which contains both violent and non-violent videos and includes six categories of anomalies: fights, shootings, protests, explosions, car accidents, and abuse. 
This dataset was then used for model fine-tuning as well as for performance evaluation on the specific task of fight detection.

## Technologies and Libraries

To successfully execute the **V-JEPA2 fine-tuning** phase, a virtual environment was created with the following setup:

- **Python:** 3.11
- **Main Libraries:**
  - `transformers==4.57.1` (with AutoVideoProcessor support)
  - `torch`
  - `torchvision`
  - `torchaudio`
  - `decord`
  - `h5py`
  - `tqdm`
  - `Pillow`

 ## Dataset

- **Dataset:** XD-Violence  
- **Source:** [Link to download](https://roc-ng.github.io/XD-Violence/))  
- **Preprocessing / Notes:**
  - The original training set contains only video-level labels (no precise temporal annotations).  
  - To enable frame-level training, each training video was manually annotated: start and end frames of each anomalous event were labeled, along with the type of anomaly.  
  - Test set annotations were refined to ensure a clear correspondence between frame intervals and anomaly categories.
 
## Model / Architecture

- **Model:** V-JEPA2 (fine-tuned for fight detection)
- **Pipeline:**
  1. **Frame Sampling:** Input videos were sampled at 5 frames per second and grouped into fixed-length clips of 64 frames (matching the input size expected by V-JEPA2).  
  2. **Clip Labeling:** Each clip was labeled as containing a fight if at least 15% of its frames depicted violent behavior. This threshold ensured consistent and reliable training data.  
  3. **Fine-tuning:** Labeled clips were fed into the V-JEPA2 model, which was fine-tuned in a fully supervised manner for the fight detection task.

- **Key Hyperparameters:**
  - Clip length: 64 frames  
  - Frame sampling rate: 5 fps  
  - Fight threshold: 15% of frames per clip
 
## Results

- **Evaluation Metrics:**
  - **AUC, AUCA, AP, APA** (for detection and temporal localization), **Precision, Recall, F1-score** (for classifier performance)
  - **Inference latency** measured to assess real-time feasibility.
    
- **Performance:**
- | Metric | Value | Description |
|--------|-------|-------------|
| APA    | 96%   | Abnormal Average Precision for rare and highly imbalanced events |
| AUC    | 0.95  | Area under the ROC curve for detection |
| AUCA   | 0.94  | Abnormal Area under the ROC curve for temporal localization |
| AP     | 0.93  | Average precision |
| Precision | 0.92 | Precision for classifier performance |
| Recall    | 0.91 | Recall for classifier performance |
| F1-score  | 0.915 | F1-score for classifier performance |
| Inference latency | 1.28 s per 64-frame clip | Measured on test hardware for real-time feasibility |


## How to Run the Project
### 1. Clips Creation and Fine-tuning of V-JEPA2

- **Build dataset clips:**  
  Process videos from the XD-Violence dataset and generate 64-frame clips sampled at 5 fps. Clips are saved in HDF5 (`.h5`) format.  
  Please, run 'python build_dataset_h5.py'

- **Fine-tune V-JEPA2:**
  Reads the clips and fine-tunes the V-JEPA2 model for fight detection.
  Please, run 'python fine_tuning_vjepa2.py'

  Pre-trained model used: vjepa2-vitg-fpc64-384-ssv2 (trained on Something-Something V2)
  Download from Hugging Face: https://huggingface.co/facebook/vjepa2-vitg-fpc64-384-ssv2

- **Evaluate the model:**
  Evaluate the fine-tuned V-JEPA2 on the XD-Violence test set and report metrics.
  Please, run 'python evaluation.py'

### 2. Feature Extraction with V-JEPA2

- **Extract features:**
  Extract features from videos using V-JEPA2 and save them in HDF5 format. No fine-tuning is performed.
  Please, run 'python feature_extraction_with_vjepa2.py'

- **Evaluate features with shallow classifiers:**
  Use the extracted features as input for simple classifiers. Open the notebook 'classifier-with-vjepa2-features.ipynb' and run cells sequentially.

### 3. Comparison with State-of-the-Art Models

  Notebooks:
  - bnwvad_script.ipynb → BNWVAD
  - hl_net_script.ipynb → HL-Net
  - ur_dmu_script.ipynb → UR-DMU

  These notebooks perform inference on the XD-Violence test set.
  Open each notebook in a compatible environment (e.g., Colab) and execute all cells sequentially.

  To run these notebooks successfully, download pre-extracted features from the XD-Violence dataset: https://roc-ng.github.io/XD-Violence/

## Note:
To successfully run this project, you need to **update the file paths in the scripts and notebooks** to match your local environment.  

> All scripts and notebooks currently contain the file paths used during development.  
> Make sure to replace them with the correct paths on your machine
