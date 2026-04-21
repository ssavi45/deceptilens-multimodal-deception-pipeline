import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from pipeline.config import MODEL_DIR, DEVICE, BATCH_SIZE, RANDOM_SEED, DOLOS_FEAT_DIR

def load_features_resnet(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ["label", "video_id"]]
    X = df[feature_cols].values
    y = df["label"].values.astype(int)
    return X, y, feature_cols

class DynamicResNetDataset(Dataset):
    def __init__(self, X_vid, X_aud, y):
        self.X_v = torch.FloatTensor(X_vid)
        self.X_a = torch.FloatTensor(X_aud)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_v[idx], self.X_a[idx], self.y[idx]

###########################################################
# DYNAMIC DEEP ARCHITECTURE
###########################################################

class ResidualBlock1D(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        return self.relu(self.dropout(self.bn2(self.fc2(self.dropout(self.relu(self.bn1(self.fc1(x))))))) + res)

class DynamicStreamEncoder(nn.Module):
    def __init__(self, input_dim: int, fusion_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock1D(fusion_dim, dropout),
            ResidualBlock1D(fusion_dim, dropout)
        )
    def forward(self, x): return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5
    def forward(self, q, kv):
        scores = (self.query(q) * self.key(kv)).sum(dim=-1, keepdim=True) / self.scale
        return torch.sigmoid(scores) * self.value(kv)

class DynamicDeceptionDetector(nn.Module):
    def __init__(self, video_dim, audio_dim, fusion_dim, dropout):
        super().__init__()
        self.video_encoder = DynamicStreamEncoder(video_dim, fusion_dim, dropout)
        self.audio_encoder = DynamicStreamEncoder(audio_dim, fusion_dim, dropout)
        
        self.cross_v2a = CrossAttention(fusion_dim)
        self.cross_a2v = CrossAttention(fusion_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim*2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, v, a):
        v_enc = self.video_encoder(v)
        a_enc = self.audio_encoder(a)
        v_attn = self.cross_v2a(v_enc, a_enc)
        a_attn = self.cross_a2v(a_enc, v_enc)
        return self.classifier(torch.cat([v_enc + v_attn, a_enc + a_attn], dim=-1))

###########################################################
# PIPELINE
###########################################################

device = torch.device(DEVICE)

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for vid, aud, lbl in loader:
            p = model(vid.to(device), aud.to(device)).argmax(dim=-1)
            preds.extend(p.cpu().numpy())
            labels.extend(lbl.numpy())
    return accuracy_score(labels, preds), f1_score(labels, preds, average='macro', zero_division=0)

if __name__ == "__main__":
    print("\n==========================================================")
    print(" ADVANCED OPTIMIZATION AND HYPERPARAMETER TUNING ")
    print("==========================================================")

    dolos_csv = os.path.join(DOLOS_FEAT_DIR, "features_resnet.csv")
    if not os.path.exists(dolos_csv):
        print("Missing required features_resnet.csv files. Check extraction!")
        exit(1)

    X, y, feature_cols = load_features_resnet(dolos_csv)
    
    video_idx = list(range(104)) + list(range(138, 650))
    audio_idx = list(range(104, 138))
    
    # Stratified Splits
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=RANDOM_SEED)

    print("\n[1/3] Pruning Irrelevant Features via Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

    # Sub-select Video
    vid_importances = [(i, importances[i]) for i in video_idx]
    vid_importances.sort(key=lambda x: x[1], reverse=True)
    top_video_idx = [x[0] for x in vid_importances[:150]] # Strip over 400 noisy video dimensions!
    
    # Sub-select Audio
    aud_importances = [(i, importances[i]) for i in audio_idx]
    aud_importances.sort(key=lambda x: x[1], reverse=True)
    top_audio_idx = [x[0] for x in aud_importances[:20]] # Keep top 20 Audio
    
    print(f"Dropped {len(video_idx)-150} Video dimensions and {len(audio_idx)-20} Audio dimensions.")

    scaler_v = StandardScaler()
    X_train_v = scaler_v.fit_transform(X_train[:, top_video_idx])
    X_val_v   = scaler_v.transform(X_val[:, top_video_idx])
    X_test_v  = scaler_v.transform(X_test[:, top_video_idx])

    scaler_a = StandardScaler()
    X_train_a = scaler_a.fit_transform(X_train[:, top_audio_idx])
    X_val_a   = scaler_a.transform(X_val[:, top_audio_idx])
    X_test_a  = scaler_a.transform(X_test[:, top_audio_idx])

    train_loader = DataLoader(DynamicResNetDataset(X_train_v, X_train_a, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(DynamicResNetDataset(X_val_v, X_val_a, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(DynamicResNetDataset(X_test_v, X_test_a, y_test), batch_size=BATCH_SIZE, shuffle=False)

    print("\n[2/3] Executing Grid Search Over Configurations...")
    grids = [
        {"lr": 3e-4, "dropout": 0.4, "fusion_dim": 256, "wd": 1e-4},
        {"lr": 1e-4, "dropout": 0.5, "fusion_dim": 256, "wd": 1e-3},
        {"lr": 5e-4, "dropout": 0.3, "fusion_dim": 128, "wd": 1e-4},
        {"lr": 1e-4, "dropout": 0.4, "fusion_dim": 512, "wd": 1e-4},
        {"lr": 3e-4, "dropout": 0.6, "fusion_dim": 256, "wd": 1e-3},
    ]

    best_v_f1 = 0
    best_config = None
    best_state_global = None

    for idx, config in enumerate(grids):
        model = DynamicDeceptionDetector(150, 20, config["fusion_dim"], config["dropout"]).to(device)
        optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        cfg_best_v_f1 = 0
        cfg_best_state = None
        
        for epoch in range(1, 41): # Train each grid option explicitly for 40 fast epochs
            model.train()
            for vid, aud, lbl in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(vid.to(device), aud.to(device)), lbl.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            
            _, v_f1 = evaluate(model, val_loader)
            if v_f1 > cfg_best_v_f1:
                cfg_best_v_f1 = v_f1
                cfg_best_state = copy.deepcopy(model.state_dict())
                
        print(f"   Config {idx+1}/{len(grids)} {config} -> Val F1: {cfg_best_v_f1:.4f}")
        if cfg_best_v_f1 > best_v_f1:
            best_v_f1 = cfg_best_v_f1
            best_config = config
            best_state_global = cfg_best_state

    print("\n[3/3] Saving Final Absolute Peak Metrics...")
    model = DynamicDeceptionDetector(150, 20, best_config["fusion_dim"], best_config["dropout"]).to(device)
    model.load_state_dict(best_state_global)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "tuned_optimal_model.pth"))
    
    t_acc, t_f1 = evaluate(model, test_loader)
    print("==========================================================")
    print("  ABSOLUTE TUNED AND REDUCED TESTING ACCURACY ")
    print("==========================================================")
    print(f"Optimal Parameters: {best_config}")
    print(f"Final DOLOS Target Base Accuracy: {t_acc:.4f} (F1: {t_f1:.4f})")
    print("==========================================================")
