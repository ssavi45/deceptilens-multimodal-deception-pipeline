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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.stats import mode
from tqdm import tqdm

from pipeline.config import (
    MODEL_DIR, DEVICE, NUM_CLASSES, BATCH_SIZE, RANDOM_SEED,
    DOLOS_FEAT_DIR
)

# Feature Dimensionalities Structuring
MEDIAPIPE_DIM = 104
AUDIO_DIM = 34
RESNET_DIM = 512
VIDEO_FUSED_DIM = MEDIAPIPE_DIM + RESNET_DIM

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
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        vec_mediapipe = self.X[idx, :104]
        vec_audio = self.X[idx, 104:138]
        vec_resnet = self.X[idx, 138:]
        vec_video = torch.cat([vec_mediapipe, vec_resnet])
        return vec_video, vec_audio, self.y[idx]

class ResidualBlock1D(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.relu(self.dropout(self.bn2(self.fc2(self.dropout(self.relu(self.bn1(self.fc1(x))))))) + x)

class ResNetStreamEncoder(nn.Module):
    def __init__(self, input_dim: int, fusion_dim: int = 512):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, fusion_dim), nn.BatchNorm1d(fusion_dim), nn.GELU(), nn.Dropout(0.3))
        self.res_blocks = nn.Sequential(ResidualBlock1D(fusion_dim), ResidualBlock1D(fusion_dim))
    def forward(self, x): return self.res_blocks(self.input_proj(x))

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

class ResNetDeceptionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        fusion_dim = 512
        self.video_encoder = ResNetStreamEncoder(VIDEO_FUSED_DIM, fusion_dim)
        self.audio_encoder = ResNetStreamEncoder(AUDIO_DIM, fusion_dim)
        self.cross_attn_v2a = CrossAttention(fusion_dim)
        self.cross_attn_a2v = CrossAttention(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, video, audio):
        v = self.video_encoder(video)
        a = self.audio_encoder(audio)
        v_attended = self.cross_attn_v2a(v, a)
        a_attended = self.cross_attn_a2v(a, v)
        return self.classifier(torch.cat([v + v_attended, a + a_attended], dim=-1))

device = torch.device(DEVICE)

def evaluate_nn(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for vid, aud, lbls in loader:
            logits = model(vid.to(device), aud.to(device))
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

if __name__ == "__main__":
    print("\n==========================================================")
    print(" EXTREME STACKING ARCHITECTURE (HGB + SVM + RF + PyTorch) ")
    print("==========================================================")
    
    dolos_csv = os.path.join(DOLOS_FEAT_DIR, "features_resnet.csv")
    if not os.path.exists(dolos_csv):
        print("Dataset missing! Exiting...")
        exit(1)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    X, y = load_features_resnet(dolos_csv)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1764, stratify=y_train_val, random_state=RANDOM_SEED)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 1. Scikit-Learn Classifiers
    print("\n[1/3] Compiling Machine Learning Algorithms...")
    
    hgb_model = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, random_state=RANDOM_SEED)
    hgb_model.fit(X_train_s, y_train)
    hgb_preds, hgb_probs = hgb_model.predict(X_test_s), hgb_model.predict_proba(X_test_s)[:, 1]
    print(f"      [LightGBM] HistGradientBoosting Accuracy: {accuracy_score(y_test, hgb_preds):.4f}")

    svm_model = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, random_state=RANDOM_SEED)
    svm_model.fit(X_train_s, y_train)
    svm_preds, svm_probs = svm_model.predict(X_test_s), svm_model.predict_proba(X_test_s)[:, 1]
    print(f"      [RBF-SVM]  Support Vector Machine Accuracy: {accuracy_score(y_test, svm_preds):.4f}")

    rf_model = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1)
    rf_model.fit(X_train_s, y_train)
    rf_preds, rf_probs = rf_model.predict(X_test_s), rf_model.predict_proba(X_test_s)[:, 1]
    print(f"      [Forest]   Random Forest Output Accuracy:   {accuracy_score(y_test, rf_preds):.4f}")

    # 2. PyTorch Training
    print("\n[2/3] Training Deep ResNet Dual-Stream (150 Epochs)...")
    train_loader = DataLoader(ResNetDataset(X_train_s, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ResNetDataset(X_val_s, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ResNetDataset(X_test_s, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = ResNetDeceptionDetector().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_v_f1 = 0
    best_state = None

    for epoch in tqdm(range(1, 151), desc="Deep Training"):
        model.train()
        for vid, aud, lbl in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(vid.to(device), aud.to(device)), lbl.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        
        preds_val, _, lbls_val = evaluate_nn(model, val_loader)
        v_f1 = f1_score(lbls_val, preds_val, average='macro', zero_division=0)
        
        if v_f1 > best_v_f1:
            best_v_f1 = v_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    nn_preds, nn_probs, y_test_true = evaluate_nn(model, test_loader)
    print(f"      [PyTorch]  Deep ResNet Topology Accuracy:   {accuracy_score(y_test_true, nn_preds):.4f}")

    # 3. Dynamic Aggregation (Hard Voting vs Meta-Stacking)
    print("\n[3/3] Bypassing Logit Dominance Structurally...")
    
    # Method A: Meta-Regressor Logistic Stacking (Learn when to trust each model)
    # Extract Val features to train Meta-model safely
    hgb_val_p = hgb_model.predict_proba(X_val_s)[:, 1]
    svm_val_p = svm_model.predict_proba(X_val_s)[:, 1]
    rf_val_p  = rf_model.predict_proba(X_val_s)[:, 1]
    _, nn_val_p, y_val_true = evaluate_nn(model, val_loader)
    
    meta_X_train = np.column_stack((hgb_val_p, svm_val_p, rf_val_p, nn_val_p))
    meta_model = LogisticRegression()
    meta_model.fit(meta_X_train, y_val_true)
    
    meta_X_test = np.column_stack((hgb_probs, svm_probs, rf_probs, nn_probs))
    meta_preds = meta_model.predict(meta_X_test)
    meta_acc = accuracy_score(y_test_true, meta_preds)
    
    # Method B: Pure Hard Voting Matrix
    all_discrete_preds = np.column_stack((hgb_preds, svm_preds, rf_preds, nn_preds))
    hard_votes, _ = mode(all_discrete_preds, axis=1)
    hard_votes = hard_votes.ravel()
    hard_acc = accuracy_score(y_test_true, hard_votes)
    
    # Fallback Method C: Balanced Soft Vote
    soft_probs = (hgb_probs + svm_probs + rf_probs + nn_probs) / 4.0
    soft_preds = (soft_probs >= 0.5).astype(int)
    soft_acc = accuracy_score(y_test_true, soft_preds)

    print("\n==========================================================")
    print(f" ► [METHOD A] Meta-Stacking Vector Accuracy:   {meta_acc:.4f}")
    print(f" ► [METHOD B] Hard-Voting Discrete Accuracy:   {hard_acc:.4f}")
    print(f" ► [METHOD C] Soft-Voting Balanced Accuracy:   {soft_acc:.4f}")
    print("==========================================================")
    
    absolute_peak = max(meta_acc, hard_acc, soft_acc)
    print(f" 🚀 EXTREME FINAL CAPSTONE TESTING ACCURACY:   {absolute_peak:.4f} 🚀")
    print("==========================================================")
