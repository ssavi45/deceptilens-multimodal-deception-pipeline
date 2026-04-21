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
from tqdm import tqdm

from pipeline.config import MODEL_DIR, DEVICE, NUM_CLASSES, BATCH_SIZE, RANDOM_SEED, DOLOS_FEAT_DIR

def load_temporal_dataset(csv_path, npy_path):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ["label", "video_id"]]
    static_X = df[feature_cols].values
    y = df["label"].values.astype(int)
    seq_X = np.load(npy_path) # [N, 15, 512]
    return static_X, seq_X, y

class TemporalTransformerDataset(Dataset):
    def __init__(self, static_X, seq_X, y):
        # MediaPipe Video Topography = indices [0:104]
        self.static_v = torch.FloatTensor(static_X[:, :104])
        # Audio Frequencies = indices [104:]
        self.static_a = torch.FloatTensor(static_X[:, 104:])
        # Deep Time-Sequence Visual Embeddings
        self.seq_v = torch.FloatTensor(seq_X) 
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): 
        return self.static_v[idx], self.static_a[idx], self.seq_v[idx], self.y[idx]

###########################################################
# SOTA TEMPORAL LSTM SEQUENCE TOPOLOGY
###########################################################

class TemporalSequenceLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=0.3, bidirectional=True)
    def forward(self, x):
        # Process the seq_length of Fractions of a Second mapping micro-expressions
        output, (hn, cn) = self.lstm(x)
        # Capture the forward/backward temporal state alignment [Batch, hidden*2] -> [Batch, 512]
        return torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)

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

class TemporalDeceptionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Temporal Sequence Parser
        self.temporal_lstm = TemporalSequenceLSTM()
        
        # 2. Main Encoders
        fusion_dim = 512
        self.video_encoder = ResNetStreamEncoder(104 + 512, fusion_dim)
        self.audio_encoder = ResNetStreamEncoder(34, fusion_dim)
        
        # 3. Cognitive Cross Attention
        self.cross_attn_v2a = CrossAttention(fusion_dim)
        self.cross_attn_a2v = CrossAttention(fusion_dim)
        
        # 4. Final MLP
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, stat_v, stat_a, seq_v):
        # Squeeze 15-frame time dimension sequentially using Bi-LSTM
        temporal_embedding = self.temporal_lstm(seq_v) # Shape: [512]
        
        # Concatenate sequential output explicitly with the static topography
        fused_video_features = torch.cat([stat_v, temporal_embedding], dim=-1)
        
        v = self.video_encoder(fused_video_features)
        a = self.audio_encoder(stat_a)
        
        v_attended = self.cross_attn_v2a(v, a)
        a_attended = self.cross_attn_a2v(a, v)
        return self.classifier(torch.cat([v + v_attended, a + a_attended], dim=-1))

device = torch.device(DEVICE)

def evaluate_nn(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for stat_v, stat_a, seq_v, lbls in loader:
            logits = model(stat_v.to(device), stat_a.to(device), seq_v.to(device))
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(lbls.numpy())
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro', zero_division=0)

if __name__ == "__main__":
    print("\n==========================================================")
    print(" SOTA TEMPORAL SEQUENCE LSTM ARCHITECTURE ")
    print("==========================================================")
    
    dolos_csv = os.path.join(DOLOS_FEAT_DIR, "features.csv")
    dolos_npy = os.path.join(DOLOS_FEAT_DIR, "resnet_temporal.npy")
    if not os.path.exists(dolos_npy):
        print("Temporal sequences missing! Let the background extraction script finish parsing.")
        exit(1)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    static_X, seq_X, y = load_temporal_dataset(dolos_csv, dolos_npy)
    
    X_s_tv, X_s_test, seq_tv, seq_test, y_tv, y_test = train_test_split(static_X, seq_X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED)
    X_s_train, X_s_val, seq_train, seq_val, y_train, y_val = train_test_split(X_s_tv, seq_tv, y_tv, test_size=0.1764, stratify=y_tv, random_state=RANDOM_SEED)
    
    # Scale static parameters
    scaler = StandardScaler()
    X_s_train = scaler.fit_transform(X_s_train)
    X_s_val = scaler.transform(X_s_val)
    X_s_test = scaler.transform(X_s_test)
    
    # Sequence Tensor Batch Normalization bounds
    seq_mean = seq_train.mean(axis=(0, 1), keepdims=True)
    seq_std = seq_train.std(axis=(0, 1), keepdims=True) + 1e-7
    seq_train = (seq_train - seq_mean) / seq_std
    seq_val = (seq_val - seq_mean) / seq_std
    seq_test = (seq_test - seq_mean) / seq_std

    train_loader = DataLoader(TemporalTransformerDataset(X_s_train, seq_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TemporalTransformerDataset(X_s_val, seq_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TemporalTransformerDataset(X_s_test, seq_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    print("\n[1/1] Executing Temporal Sequence Sequence Over 150 Deep Epochs...")
    model = TemporalDeceptionDetector().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # Optimized decay constraints
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_v_f1 = 0
    best_state = None

    for epoch in tqdm(range(1, 151), desc="LSTM Sequencing"):
        model.train()
        for stat_v, stat_a, seq_v, lbl in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(stat_v.to(device), stat_a.to(device), seq_v.to(device)), lbl.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        
        _, v_f1 = evaluate_nn(model, val_loader)
        if v_f1 > best_v_f1:
            best_v_f1 = v_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate_nn(model, test_loader)
    
    print("\n==========================================================")
    print(f" TEMPORAL MICRO-EXPRESSION CAPSTONE ACCURACY: {test_acc:.4f} 🚀")
    print(f"                                Final F1 Sum: {test_f1:.4f}")
    print("==========================================================")
