import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pipeline.config import (
    MODEL_DIR, DEVICE, NUM_CLASSES, BATCH_SIZE, RANDOM_SEED,
    DOLOS_FEAT_DIR, TRIAL_FEAT_DIR
)

# Feature Dimensionalities Structuring
MEDIAPIPE_DIM = 104
AUDIO_DIM = 34
RESNET_DIM = 512
VIDEO_FUSED_DIM = MEDIAPIPE_DIM + RESNET_DIM  # 616

###########################################################
# CUSTOM RESNET DATALOADER
###########################################################

def load_features_resnet(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ["label", "video_id"]]
    X = df[feature_cols].values
    y = df["label"].values.astype(int)
    return X, y

class ResNetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        # Due to pandas appending logic: 
        # [0:104] is MediaPipe | [104:138] is Audio | [138:650] is ResNet
        vec_mediapipe = self.X[idx, :104]
        vec_audio = self.X[idx, 104:138]
        vec_resnet = self.X[idx, 138:]
        
        vec_video = torch.cat([vec_mediapipe, vec_resnet])
        return vec_video, vec_audio, self.y[idx]

###########################################################
# DEEP RESNET ARCHITECTURE
###########################################################

class ResidualBlock1D(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.dropout(self.relu(self.bn1(self.fc1(x))))
        out = self.dropout(self.bn2(self.fc2(out)))
        return self.relu(out + residual)

class ResNetStreamEncoder(nn.Module):
    def __init__(self, input_dim: int, fusion_dim: int = 512):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock1D(fusion_dim),
            ResidualBlock1D(fusion_dim)
        )

    def forward(self, x):
        return self.res_blocks(self.input_proj(x))

class CrossAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, q_in, kv_in):
        Q = self.query(q_in)
        K = self.key(kv_in)
        V = self.value(kv_in)
        attn_scores = (Q * K).sum(dim=-1, keepdim=True) / self.scale
        attn_weights = torch.sigmoid(attn_scores)
        return attn_weights * V

class ResNetDeceptionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        fusion_dim = 256 # Tuned optimal parameter (down from 512)
        self.video_encoder = ResNetStreamEncoder(VIDEO_FUSED_DIM, fusion_dim)
        self.audio_encoder = ResNetStreamEncoder(AUDIO_DIM, fusion_dim)
        
        self.cross_attn_v2a = CrossAttention(fusion_dim)
        self.cross_attn_a2v = CrossAttention(fusion_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4), # Tuned optimal dropout
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, video, audio):
        v = self.video_encoder(video)
        a = self.audio_encoder(audio)
        v_attended = self.cross_attn_v2a(v, a)
        a_attended = self.cross_attn_a2v(a, v)
        return self.classifier(torch.cat([v + v_attended, a + a_attended], dim=-1))

###########################################################
# TRAINING HELPER LOGIC
###########################################################

device = torch.device(DEVICE)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for vid, aud, lbls in loader:
            logits = model(vid.to(device), aud.to(device))
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro", zero_division=0), auc

def train_resnet_model(feat_csv, run_name, epochs=150):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    X, y = load_features_resnet(feat_csv)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1764, stratify=y_train_val, random_state=RANDOM_SEED)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    train_loader = DataLoader(ResNetDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ResNetDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ResNetDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = ResNetDeceptionDetector().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4) # Tuned optimal LR and WD
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_v_f1 = 0
    best_state = None

    print(f"\n--- Training Deep Visual {run_name.upper()} Model ({epochs} Epochs) ---")
    from tqdm import tqdm
    for epoch in tqdm(range(1, epochs + 1), desc=f"Training {run_name.upper()}"):
        model.train()
        for vid, aud, lbl in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(vid.to(device), aud.to(device)), lbl.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        
        _, v_f1, _ = evaluate(model, val_loader)
        if v_f1 > best_v_f1:
            best_v_f1 = v_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{run_name}_resnet.pth"))
    
    t_acc, t_f1, t_auc = evaluate(model, test_loader)
    print(f"\n[{run_name} Baseline] Accuracy: {t_acc:.4f} | F1: {t_f1:.4f} | AUC: {t_auc:.4f}")

def protocol_zero_shot(src_csv, tgt_csv, src_name, tgt_name):
    mpath = os.path.join(MODEL_DIR, f"{src_name.lower()}_resnet.pth")
    model = ResNetDeceptionDetector().to(device)
    model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
    
    X_target, y_target = load_features_resnet(tgt_csv)
    # Use standard scaling instead of re-fitting test features
    scaler = StandardScaler()
    X_target = scaler.fit_transform(X_target)
    
    loader = DataLoader(ResNetDataset(X_target, y_target), batch_size=BATCH_SIZE, shuffle=False)
    acc, f1, auc = evaluate(model, loader)
    print(f"[Protocol A: {src_name}->{tgt_name}] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

def protocol_few_shot(src_csv, tgt_csv, src_name, tgt_name):
    mpath = os.path.join(MODEL_DIR, f"{src_name.lower()}_resnet.pth")
    model = ResNetDeceptionDetector().to(device)
    model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
    
    # Freeze Encoders
    for p in model.video_encoder.parameters(): p.requires_grad = False
    for p in model.audio_encoder.parameters(): p.requires_grad = False
    for p in model.cross_attn_v2a.parameters(): p.requires_grad = False
    for p in model.cross_attn_a2v.parameters(): p.requires_grad = False

    X, y = load_features_resnet(tgt_csv)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Take 10% for few-shot, rest for test
    X_few, X_test, y_few, y_test = train_test_split(X, y, train_size=0.10, stratify=y, random_state=RANDOM_SEED)
    few_loader = DataLoader(ResNetDataset(X_few, y_few), batch_size=8, shuffle=True)
    test_loader = DataLoader(ResNetDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        model.train()
        for vid, aud, lbl in few_loader:
            optimizer.zero_grad()
            loss = criterion(model(vid.to(device), aud.to(device)), lbl.to(device))
            loss.backward()
            optimizer.step()

    acc, f1, auc = evaluate(model, test_loader)
    print(f"[Protocol B: {src_name}->{tgt_name} Few-Shot] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

if __name__ == "__main__":
    dolos_csv = os.path.join(DOLOS_FEAT_DIR, "features_resnet.csv")
    trial_csv = os.path.join(TRIAL_FEAT_DIR, "features_resnet.csv")
    
    if not os.path.isfile(dolos_csv):
        print("Missing DOLOS features_resnet.csv! Please run extract_resnet18.py first.")
        exit(1)

    print("==========================================================")
    print(" DEEP VISUAL EMBEDDINGS PROTOCOLS (RESNET-18) ")
    print("==========================================================")
    train_resnet_model(dolos_csv, "dolos")
    if os.path.exists(trial_csv):
        train_resnet_model(trial_csv, "trial")
        
        print("\n--- ZERO-SHOT TRANSFER (PROTOCOL A) ---")
        protocol_zero_shot(dolos_csv, trial_csv, "DOLOS", "Trial")
        protocol_zero_shot(trial_csv, dolos_csv, "Trial", "DOLOS")

        print("\n--- FEW-SHOT TRANSFER (PROTOCOL B) ---")
        protocol_few_shot(dolos_csv, trial_csv, "DOLOS", "Trial")
        print("\n==========================================================")
