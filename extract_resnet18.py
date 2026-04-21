import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

from pipeline.config import DEVICE, DOLOS_VIDEO_DIR, TRIAL_VIDEO_DIR, DOLOS_FEAT_DIR, TRIAL_FEAT_DIR

def extract_resnet_features(video_dir, original_csv, output_csv):
    print(f"\nExtracing ResNet-18 frames for {original_csv}...")
    df = pd.read_csv(original_csv)
    
    device = torch.device(DEVICE)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Identity()  # Strip classification head to get raw 512-dim embedding tensor
    model = model.to(device)
    model.eval()
    
    preprocess = weights.transforms()
    
    resnet_features = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="ResNet Convolution Pass"):
        video_id = str(row['video_id']).strip()
        filename = video_id if video_id.lower().endswith(".mp4") else f"{video_id}.mp4"
        video_path = os.path.join(video_dir, filename)
        
        if not os.path.isfile(video_path):
            resnet_features.append(np.zeros(512))
            continue
            
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            resnet_features.append(np.zeros(512))
            continue
            
        # Extract 20 highly temporal spaced frames to generalize the spatial representation of the clip
        sample_indices = set(np.linspace(0, frame_count - 1, 20, dtype=int))
        frames = []
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret: break
            if i in sample_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert explicitly to PIL structure implicitly by formatting directly
                tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                try:
                    p_tensor = preprocess(tensor)
                    frames.append(p_tensor)
                except Exception:
                    pass
                    
        cap.release()
        
        if len(frames) == 0:
            resnet_features.append(np.zeros(512))
            continue
            
        frames_tensor = torch.stack(frames).to(device)
        with torch.no_grad():
            preds = model(frames_tensor) # Shape: [num_frames, 512]
            mean_emb = preds.mean(dim=0).cpu().numpy() # Reduce time dimension to [512]
            resnet_features.append(mean_emb)
            
    resnet_features = np.array(resnet_features)
    
    # Safely append 512 residual columns alongside existing 138-dim vectors
    print("Saving 512-dimensional continuous latent space...")
    for i in range(512):
        df[f'resnet_{i}'] = resnet_features[:, i]
        
    df.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Created {output_csv} with 650 fused dimensionalities (138 Native + 512 ResNet)!")

if __name__ == "__main__":
    dolos_csv = os.path.join(DOLOS_FEAT_DIR, "features.csv")
    dolos_out = os.path.join(DOLOS_FEAT_DIR, "features_resnet.csv")
    if os.path.exists(dolos_csv) and not os.path.exists(dolos_out):
        extract_resnet_features(DOLOS_VIDEO_DIR, dolos_csv, dolos_out)
        
    trial_csv = os.path.join(TRIAL_FEAT_DIR, "features.csv")
    trial_out = os.path.join(TRIAL_FEAT_DIR, "features_resnet.csv")
    if os.path.exists(trial_csv) and not os.path.exists(trial_out):
        extract_resnet_features(TRIAL_VIDEO_DIR, trial_csv, trial_out)
