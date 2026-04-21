import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import logging

from pipeline.config import (
    MODEL_DIR, DEVICE, VIDEO_INPUT_DIM, AUDIO_INPUT_DIM, 
    NUM_CLASSES, DROPOUT, BATCH_SIZE, RANDOM_SEED,
    DOLOS_FEAT_DIR, TRIAL_FEAT_DIR
)
from pipeline.dataset import load_features, split_and_normalize, make_loader, prepare_cross_dataset

###########################################################
# ADVANCED RESIDUAL ARCHITECTURE
###########################################################

class ResidualBlock1D(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return self.relu(out + residual)

class AdvancedStreamEncoder(nn.Module):
    def __init__(self, input_dim: int, fusion_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock1D(fusion_dim, dropout),
            ResidualBlock1D(fusion_dim, dropout)
        )

    def forward(self, x):
        x = self.input_proj(x)
        return self.res_blocks(x)

class CrossAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, query_input, kv_input):
        Q = self.query(query_input)
        K = self.key(kv_input)
        V = self.value(kv_input)
        attn_scores = (Q * K).sum(dim=-1, keepdim=True) / self.scale
        attn_weights = torch.sigmoid(attn_scores)
        attended = attn_weights * V
        return attended

class AdvancedDeceptionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        fusion_dim = 256
        self.video_encoder = AdvancedStreamEncoder(VIDEO_INPUT_DIM, fusion_dim)
        self.audio_encoder = AdvancedStreamEncoder(AUDIO_INPUT_DIM, fusion_dim)
        self.cross_attn_v2a = CrossAttention(fusion_dim)
        self.cross_attn_a2v = CrossAttention(fusion_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, video, audio):
        v_enc = self.video_encoder(video)
        a_enc = self.audio_encoder(audio)
        v_attended = self.cross_attn_v2a(v_enc, a_enc)
        a_attended = self.cross_attn_a2v(a_enc, v_enc)
        fused = torch.cat([v_enc + v_attended, a_enc + a_attended], dim=-1)
        return self.classifier(fused)

###########################################################
# TRAINING & EVALUATION HELPER LOGIC
###########################################################

device = torch.device(DEVICE)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for video, audio, labels in loader:
            logits = model(video.to(device), audio.to(device))
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    return acc, f1, auc

def train_advanced(feat_csv, run_name, epochs=150):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    X, y = load_features(feat_csv)
    data = split_and_normalize(X, y, fit_scaler=True, scaler_save_path=os.path.join(MODEL_DIR, f"{run_name}_scaler.pkl"))
    
    train_loader = make_loader(data["X_train"], data["y_train"], BATCH_SIZE, shuffle=True, balance=True)
    val_loader = make_loader(data["X_val"], data["y_val"], BATCH_SIZE, shuffle=False)
    test_loader = make_loader(data["X_test"], data["y_test"], BATCH_SIZE, shuffle=False)

    model = AdvancedDeceptionDetector().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_f1 = 0
    best_state = None

    print(f"\n--- Training {run_name.upper()} Model ({epochs} Epochs) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        for video, audio, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(video.to(device), audio.to(device)), labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        
        _, v_f1, _ = evaluate(model, val_loader)
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model_save_path = os.path.join(MODEL_DIR, f"{run_name}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    
    t_acc, t_f1, t_auc = evaluate(model, test_loader)
    print(f"[{run_name} Baseline] Accuracy: {t_acc:.4f} | F1: {t_f1:.4f} | AUC: {t_auc:.4f}")
    return model

def protocol_zero_shot(src_csv, tgt_csv, src_name, tgt_name):
    mpath = os.path.join(MODEL_DIR, f"{src_name.lower()}_model.pth")
    spath = os.path.join(MODEL_DIR, f"{src_name.lower()}_scaler.pkl")
    
    model = AdvancedDeceptionDetector().to(device)
    model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
    
    data = prepare_cross_dataset(src_csv, tgt_csv, few_shot_frac=0.0, source_scaler_path=spath)
    loader = make_loader(data["X_target"], data["y_target"], BATCH_SIZE, shuffle=False)
    
    acc, f1, auc = evaluate(model, loader)
    print(f"[Protocol A: {src_name}->{tgt_name}] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

def protocol_few_shot(src_csv, tgt_csv, src_name, tgt_name):
    mpath = os.path.join(MODEL_DIR, f"{src_name.lower()}_model.pth")
    spath = os.path.join(MODEL_DIR, f"{src_name.lower()}_scaler.pkl")
    
    model = AdvancedDeceptionDetector().to(device)
    model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
    
    # Freeze Encoders
    for p in model.video_encoder.parameters(): p.requires_grad = False
    for p in model.audio_encoder.parameters(): p.requires_grad = False
    for p in model.cross_attn_v2a.parameters(): p.requires_grad = False
    for p in model.cross_attn_a2v.parameters(): p.requires_grad = False

    data = prepare_cross_dataset(src_csv, tgt_csv, few_shot_frac=0.1, source_scaler_path=spath)
    few_loader = make_loader(data["X_few"], data["y_few"], batch_size=8, shuffle=True, balance=True)
    test_loader = make_loader(data["X_target"], data["y_target"], BATCH_SIZE, shuffle=False)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        model.train()
        for video, audio, labels in few_loader:
            optimizer.zero_grad()
            loss = criterion(model(video.to(device), audio.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()

    acc, f1, auc = evaluate(model, test_loader)
    print(f"[Protocol B: {src_name}->{tgt_name} Few-Shot] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

if __name__ == "__main__":
    dolos_csv = os.path.join(DOLOS_FEAT_DIR, "features.csv")
    trial_csv = os.path.join(TRIAL_FEAT_DIR, "features.csv")
    
    if not os.path.isfile(dolos_csv):
        print("Missing DOLOS features cache. Run the main pipeline first!")
        exit(1)

    print("==========================================================")
    print(" ADVANCED ARCHITECTURE EXPERIMENTS ")
    print("==========================================================")

    # PROTOCOL C: Train baselines
    train_advanced(dolos_csv, "dolos")
    train_advanced(trial_csv, "trial")

    print("\n--- ZERO-SHOT TRANSFER (PROTOCOL A) ---")
    protocol_zero_shot(dolos_csv, trial_csv, "DOLOS", "Trial")
    protocol_zero_shot(trial_csv, dolos_csv, "Trial", "DOLOS")

    print("\n--- FEW-SHOT TRANSFER (PROTOCOL B) ---")
    protocol_few_shot(dolos_csv, trial_csv, "DOLOS", "Trial")

    print("\n==========================================================")
    print(" EVALUATION COMPLETE")
    print("==========================================================")
