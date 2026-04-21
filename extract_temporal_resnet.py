import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

from pipeline.config import DEVICE, DOLOS_VIDEO_DIR, TRIAL_VIDEO_DIR, DOLOS_FEAT_DIR, TRIAL_FEAT_DIR

def extract_resnet_time_sequence(video_dir, original_csv, out_npy, out_meta):
    print(f"\n[Temporal Extract] Processing Sequence Data for {original_csv}...")
    df = pd.read_csv(original_csv)
    
    device = torch.device(DEVICE)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Identity()  # Strip classification head -> 512-dim embedding
    model = model.to(device)
    model.eval()
    
    preprocess = weights.transforms()
    
    all_sequences = [] # target shape will be [N, 15, 512]
    meta_records = []
    
    seq_length = 15 # Precisely 15 frames to capture temporal micro-expression shifting
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Temporal Convolution Pass"):
        video_id = str(row['video_id']).strip()
        filename = f"{video_id}.mp4" if not video_id.lower().endswith(".mp4") else video_id
        video_path = os.path.join(video_dir, filename)
        
        # Meta tracking
        meta_records.append({"video_id": video_id, "label": row["label"]})
        
        if not os.path.isfile(video_path):
            all_sequences.append(np.zeros((seq_length, 512)))
            continue
            
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            all_sequences.append(np.zeros((seq_length, 512)))
            continue
            
        sample_indices = np.linspace(0, frame_count - 1, seq_length, dtype=int)
        frames = []
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret: break
            if i in sample_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                try:
                    p_tensor = preprocess(tensor)
                    frames.append(p_tensor)
                except Exception:
                    pass
                    
        cap.release()
        
        # Padding bounds if frames got dropped
        while len(frames) < seq_length:
            if len(frames) == 0:
                frames.append(torch.zeros(3, 224, 224))
            else:
                frames.append(frames[-1])
        frames = frames[:seq_length] # Hard cap at exactly 15 sequential dimensions
            
        frames_tensor = torch.stack(frames).to(device)
        with torch.no_grad():
            preds = model(frames_tensor) # Shape: [15, 512]
            all_sequences.append(preds.cpu().numpy())
            
    final_sequence_matrix = np.array(all_sequences) # [N, 15, 512]
    np.save(out_npy, final_sequence_matrix)
    
    meta_df = pd.DataFrame(meta_records)
    meta_df.to_csv(out_meta, index=False)
    
    print(f"[SUCCESS] Sequence Matrix constructed with dimensionality {final_sequence_matrix.shape} -> saved to {out_npy}!")

if __name__ == "__main__":
    dolos_csv = os.path.join(DOLOS_FEAT_DIR, "features.csv") # Read the master meta files
    dolos_out_npy = os.path.join(DOLOS_FEAT_DIR, "resnet_temporal.npy")
    dolos_out_meta = os.path.join(DOLOS_FEAT_DIR, "resnet_temporal_meta.csv")
    if os.path.exists(dolos_csv) and not os.path.exists(dolos_out_npy):
        extract_resnet_time_sequence(DOLOS_VIDEO_DIR, dolos_csv, dolos_out_npy, dolos_out_meta)
        
    trial_csv = os.path.join(TRIAL_FEAT_DIR, "features.csv")
    trial_out_npy = os.path.join(TRIAL_FEAT_DIR, "resnet_temporal.npy")
    trial_out_meta = os.path.join(TRIAL_FEAT_DIR, "resnet_temporal_meta.csv")
    if os.path.exists(trial_csv) and not os.path.exists(trial_out_npy):
        extract_resnet_time_sequence(TRIAL_VIDEO_DIR, trial_csv, trial_out_npy, trial_out_meta)
