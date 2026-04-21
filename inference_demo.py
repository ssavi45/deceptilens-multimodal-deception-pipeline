import sys
import numpy as np
import torch
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from pipeline.config import DEVICE, MODEL_PATH, SCALER_PATH, FEAT_NAMES_PATH
from pipeline.model import DeceptionDetector
from pipeline.feature_extraction import (
    extract_video_clip_features, 
    extract_audio_from_video, 
    extract_audio_features
)
from pipeline.xai import explain_single_sample, plot_single_sample_explanation

def analyze_video(video_path: str):
    print(f"\n[1/3] Extracting MediaPipe + Audio features for: {video_path}...")
    
    vid_feat = extract_video_clip_features(video_path, "tmp")
    if vid_feat is None:
        print("Error extracting video features.")
        return
        
    wav_path = extract_audio_from_video(video_path, "tmp")
    aud_feat = extract_audio_features(wav_path)
    if aud_feat is None:
        print("Error extracting audio features.")
        return
        
    # Combine into 138-dim vector
    vec = np.concatenate([vid_feat, aud_feat]).astype(np.float32)
    
    print("[2/3] Loading Model Weights & Scalers...")
    device = torch.device(DEVICE)
    model = DeceptionDetector().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    scaler = joblib.load(SCALER_PATH)
    vec_scaled = scaler.transform([vec])
    
    with open(FEAT_NAMES_PATH) as f:
        feat_names = json.load(f)
        
    print("[3/3] Running Inference & Explaining logic...")
    explanation = explain_single_sample(vec_scaled[0], model, feat_names, device, top_k=5)
    
    print("\n" + "="*50)
    print("  FINAL PREDICTION RESULTS")
    print("="*50)
    print(f"• Classifier Guess: {explanation['prediction'].upper()}")
    print(f"• Confidence Score: {explanation['confidence']:.1%}")
    print(f"\n• Model Reasoning:")
    print(f"  {explanation['explanation_text']}")
    print("="*50)
    
    save_img = "single_clip_breakdown.png"
    plot_single_sample_explanation(explanation, save_img)
    print(f"\n✓ Saved exact visual reasoning chart to: {save_img}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_demo.py <path_to_video.mp4>")
    else:
        analyze_video(sys.argv[1])
