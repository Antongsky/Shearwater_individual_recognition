#!/usr/bin/env python3
"""
BirdNet 1024-d Feature CNN Pipeline — Individual Bird Recognition
==================================================================

Input:  BirdNet feature embeddings pre-extracted as CSV
        Required columns: file_path, start, end, embedding
        (embedding column = comma-separated 1024 float values per row)

Architecture:
  1D-CNN  [B, 1, 1024] → Conv blocks → GlobalAvgPool → FC → [B, 256]
  + Supervised Contrastive Loss (SupCon) for metric learning

Why SupCon instead of ArcFace
------------------------------
ArcFace learns fixed class centres for the training individuals only.
Embeddings of unseen individuals have no guaranteed metric structure —
they land wherever the network pushes them, making cosine distance
unreliable for downstream clustering or retrieval.

SupCon directly optimises the embedding space: clips from the SAME
individual are pulled together, clips from DIFFERENT individuals are
pushed apart, with NO fixed class centres.  The resulting distance
generalises to individuals not seen at training time.

Reference: Khosla et al., "Supervised Contrastive Learning",
           NeurIPS 2020.  https://arxiv.org/abs/2004.11362

Modes:
  --train     : Train CNN on labelled embeddings, save checkpoint
  --test      : Predict individual ID for unlabelled samples
                (cosine nearest-neighbour against per-class prototypes)
  --embedding : Extract final 256-d vectors and save as .npy

CSV formats
-----------
Training CSV  (--embeddings_csv):  file_path, start, end, embedding, individual_id
              OR provide separate  --labels_csv: file_path, individual_id
              (merged automatically on file_path, with basename fallback)

Test / Embedding CSV (--embeddings_csv): file_path, start, end, embedding
              (individual_id column not required)

Multi-segment handling
----------------------
If a file has multiple rows (segments), they are aggregated PER FILE:
  - train     : every segment is its own training sample
  - test      : majority vote across all segments of the same file
  - embedding : mean-pool 256-d vectors across all segments of the same file
"""

import os
import argparse
import ntpath
import posixpath
import random
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
INPUT_DIM      = 1024     # BirdNet feature embedding dimension
EMBEDDING_DIM  = 256      # Target embedding dimension
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS    = 0        # Keep 0 for Windows compatibility
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
EPOCHS         = 50
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# SupCon hyperparameters
SUPCON_TEMP      = 0.07   # temperature τ — lower = harder negatives / tighter clusters
SUPCON_N_CLASSES = 8      # number of individuals sampled per balanced batch
SUPCON_N_SAMPLES = 4      # number of segments sampled per individual per batch


# ─────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────
def _robust_basename(p: str) -> str:
    """Return filename only, handling both Windows (\\) and POSIX (/) paths on any OS."""
    p = str(p)
    return ntpath.basename(p) if '\\' in p else posixpath.basename(p)


def load_embeddings_csv(csv_path: str, labels_csv: str = None) -> pd.DataFrame:
    """
    Load a BirdNet feature-embedding CSV.

    The 'embedding' column holds a comma-separated string of 1024 floats.
    If labels_csv is provided it is merged on 'file_path' to attach
    individual_id to every segment row.  When exact path matching fails
    (e.g. absolute Windows paths vs. bare filenames) the merge falls back
    to matching on basename only.

    Returns a DataFrame with columns:
      file_path, start, end, embedding_vec (np.ndarray shape [1024]),
      individual_id (str, may be '' if not provided)
    """
    df = pd.read_csv(csv_path)
    required = {'file_path', 'start', 'end', 'embedding'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Embeddings CSV missing columns: {missing}")

    def parse_emb(s):
        arr = np.fromstring(s, sep=',', dtype=np.float32)
        if arr.shape[0] != INPUT_DIM:
            raise ValueError(f"Expected {INPUT_DIM}-d embedding, got {arr.shape[0]}")
        return arr

    print(f"Parsing {len(df)} embedding rows …")
    df['embedding_vec'] = df['embedding'].apply(parse_emb)

    # Attach labels
    if labels_csv is not None:
        ldf = pd.read_csv(labels_csv)
        if 'individual_id' not in ldf.columns:
            raise ValueError("labels_csv must have an 'individual_id' column.")

        # Auto-detect which column in labels_csv holds the file path
        path_col_candidates = ['file_path', 'filepath', 'filename', 'path']
        ldf_path_col = None
        for c in path_col_candidates:
            if c in ldf.columns:
                ldf_path_col = c
                break
        if ldf_path_col is None:
            other_cols = [c for c in ldf.columns if c != 'individual_id']
            if not other_cols:
                raise ValueError(
                    "labels_csv must have a file path column and an 'individual_id' column."
                )
            ldf_path_col = other_cols[0]
            print(f"[INFO] labels_csv: using column '{ldf_path_col}' as file path.")
        ldf = ldf.rename(columns={ldf_path_col: 'file_path'})

        # Attempt 1: exact path match
        merged  = df.merge(ldf[['file_path', 'individual_id']], on='file_path', how='left')
        matched = int(merged['individual_id'].notna().sum())
        print(f"[INFO] Label merge (exact path): {matched}/{len(df)} rows matched.")

        # Attempt 2: basename match (handles absolute vs relative, Windows vs POSIX)
        if matched < len(df):
            df['_basename']  = df['file_path'].apply(_robust_basename)
            ldf['_basename'] = ldf['file_path'].apply(_robust_basename)

            if ldf['_basename'].duplicated().any():
                dups = ldf[ldf['_basename'].duplicated(keep=False)]['_basename'].unique()
                print(f"[WARNING] Duplicate basenames in labels_csv: {dups}. "
                      "Using first match.")

            ldf_base = ldf[['_basename', 'individual_id']].drop_duplicates('_basename')
            merged2  = df.merge(ldf_base, on='_basename', how='left')
            matched2 = int(merged2['individual_id'].notna().sum())
            print(f"[INFO] Label merge (basename):   {matched2}/{len(df)} rows matched.")

            if matched2 > matched:
                merged  = merged2
                matched = matched2

            df.drop(columns=['_basename'], inplace=True, errors='ignore')

        if matched == 0:
            print("\n[ERROR] No rows could be matched between embeddings CSV and labels CSV.")
            print(f"  Embedding CSV paths (sample): {df['file_path'].iloc[:3].tolist()}")
            print(f"  Labels CSV paths   (sample): {ldf['file_path'].iloc[:3].tolist()}")
            raise ValueError(
                "Label merge produced 0 matches. "
                "The 'file_path' values in your embeddings CSV and labels CSV do not overlap. "
                "Check that both CSVs refer to the same audio files."
            )

        if matched < len(df):
            unmatched = merged[merged['individual_id'].isna()]['file_path'].tolist()
            print(f"[WARNING] {len(unmatched)} rows could not be matched to a label "
                  f"and will be dropped:\n  {unmatched[:5]}"
                  + (" …" if len(unmatched) > 5 else ""))
            merged = merged[merged['individual_id'].notna()].reset_index(drop=True)

        df = merged
        df.drop(columns=['_basename'], inplace=True, errors='ignore')

    if 'individual_id' not in df.columns:
        df['individual_id'] = ''
    df['individual_id'] = df['individual_id'].fillna('').astype(str)
    return df


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    """One sample per CSV row (segment).  Requires individual_id to be set."""

    def __init__(self, df: pd.DataFrame, label_encoder: LabelEncoder = None):
        if df['individual_id'].eq('').any():
            raise ValueError(
                "Some rows have empty individual_id. "
                "Provide labels via --labels_csv or add an individual_id column."
            )
        self.vecs = np.stack(df['embedding_vec'].values)   # [N, 1024]

        if label_encoder is None:
            self.le     = LabelEncoder()
            self.labels = self.le.fit_transform(df['individual_id'].tolist())
        else:
            self.le     = label_encoder
            self.labels = self.le.transform(df['individual_id'].tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.vecs[idx]).float()   # [1024]
        y = int(self.labels[idx])
        return x, y


# ─────────────────────────────────────────────────────────────
# Balanced batch sampler  (required for SupCon)
# ─────────────────────────────────────────────────────────────
class BalancedBatchSampler(Sampler):
    """
    Yields batches of exactly n_classes × n_samples indices so that every
    class in the batch is represented at least n_samples times.

    This is essential for SupCon: if a class has only one representative in
    a batch there are no within-class positive pairs, and it contributes
    nothing to the contrastive loss.
    """

    def __init__(self, labels, n_classes: int = SUPCON_N_CLASSES,
                 n_samples: int = SUPCON_N_SAMPLES):
        self.labels        = np.array(labels)
        self.n_classes     = n_classes
        self.n_samples     = n_samples
        self.batch_size    = n_classes * n_samples

        self.label_to_idxs = defaultdict(list)
        for i, lbl in enumerate(self.labels):
            self.label_to_idxs[lbl].append(i)

        self.valid_classes = [
            lbl for lbl, idxs in self.label_to_idxs.items()
            if len(idxs) >= n_samples
        ]
        if len(self.valid_classes) < 2:
            raise ValueError(
                f"BalancedBatchSampler needs ≥2 classes with ≥{n_samples} samples each. "
                f"Found {len(self.valid_classes)} qualifying class(es). "
                f"Reduce --supcon_n_samples or provide more training data."
            )
        self.n_batches = max(1, len(self.labels) // self.batch_size)

    def __iter__(self):
        for _ in range(self.n_batches):
            chosen = random.sample(
                self.valid_classes,
                min(self.n_classes, len(self.valid_classes))
            )
            batch = []
            for cls in chosen:
                batch += random.choices(self.label_to_idxs[cls], k=self.n_samples)
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# ─────────────────────────────────────────────────────────────
# Feature augmentation
# ─────────────────────────────────────────────────────────────
def augment_vec(x: torch.Tensor, noise_std: float = 0.02) -> torch.Tensor:
    """
    Lightweight stochastic augmentation for 1024-d BirdNet feature vectors.
    Applied independently to each of the two SupCon views:
      - Additive Gaussian noise
      - Random 10 % dimension dropout (set to 0)
    """
    x = x + torch.randn_like(x) * noise_std
    x = x * (torch.rand_like(x) > 0.10).float()
    return x


# ─────────────────────────────────────────────────────────────
# Model — 1-D CNN on 1024-d feature vector
# ─────────────────────────────────────────────────────────────
class ConvBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class BirdCNN(nn.Module):
    """
    1-D CNN that maps a 1024-d BirdNet feature vector to a 256-d identity embedding.

    Architecture:
      [B, 1024]
        → reshape [B, 1, 1024]          (1 channel, length 1024)
        → ConvBlock(1,  64,  k=8, s=4)  → [B, 64,  255]
        → ConvBlock(64, 128, k=4, s=2)  → [B, 128, 127]
        → ConvBlock(128,256, k=4, s=2)  → [B, 256, 63]
        → ConvBlock(256,256, k=4, s=2)  → [B, 256, 31]
        → AdaptiveAvgPool1d(1)          → [B, 256]
        → Dropout(0.3)
        → FC(256, emb_dim)
        → BatchNorm1d(emb_dim)
        → L2-normalise
    """

    def __init__(self, emb_dim: int = EMBEDDING_DIM, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock1d(1,   64,  kernel=8, stride=4, padding=2),
            ConvBlock1d(64,  128, kernel=4, stride=2, padding=1),
            ConvBlock1d(128, 256, kernel=4, stride=2, padding=1),
            ConvBlock1d(256, 256, kernel=4, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(1),   # [B, 256, 1]
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1024]
        returns embedding : [B, emb_dim]  (L2-normalised)
        """
        x = x.unsqueeze(1)                 # [B, 1, 1024]
        x = self.encoder(x).squeeze(-1)    # [B, 256]
        x = self.head(x)                   # [B, emb_dim]
        x = F.normalize(x, p=2, dim=1)    # L2 normalise
        return x


# ─────────────────────────────────────────────────────────────
# Supervised Contrastive Loss
# ─────────────────────────────────────────────────────────────
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss — Khosla et al., NeurIPS 2020.
    https://arxiv.org/abs/2004.11362

    For each anchor in the batch, positives are all other embeddings sharing
    the same label; negatives are all embeddings with a different label.
    The loss maximises similarity to positives relative to negatives in
    angular (cosine) space.

    Unlike ArcFace there are no per-class weight vectors, so the learned
    metric generalises to individuals not present at training time.

    Two augmented views are created per sample during training (see
    train_one_epoch), doubling the number of positive pairs per batch.
    """

    def __init__(self, temperature: float = SUPCON_TEMP):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        emb    : [B, emb_dim]  (L2-normalised)
        labels : [B]           (long integer class indices)
        Returns scalar loss.
        """
        device = emb.device
        B = emb.shape[0]

        # Pairwise cosine similarity, scaled by temperature  [B, B]
        sim = torch.mm(emb, emb.T) / self.temperature

        # Diagonal = self-similarity → exclude from loss
        mask_self = torch.eye(B, dtype=torch.bool, device=device)

        # Positive mask: same label, different index
        labels_col = labels.contiguous().view(-1, 1)
        mask_pos   = torch.eq(labels_col, labels_col.T).float()   # [B, B]
        mask_pos[mask_self] = 0.0

        # Only compute loss for anchors that have ≥1 positive in the batch
        has_pos = mask_pos.sum(dim=1) > 0

        # Numerical stability: subtract row max (log-sum-exp trick)
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # Exponentiate; zero out self-pairs in the denominator
        exp_sim = torch.exp(sim)
        # FIXED: Use masking instead of in-place assignment
        exp_sim = exp_sim * (~mask_self).float()  # ← Non-in-place operation

        # Log-probability for every pair
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-probability over positives for each anchor
        n_pos         = mask_pos.sum(dim=1).clamp(min=1)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / n_pos

        # Average over anchors that had at least one positive partner
        loss = -mean_log_prob[has_pos].mean()
        return loss


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, supcon_loss, loader, optimizer, device,
                    noise_std: float = 0.02):
    """
    One SupCon training epoch.

    For each batch we create TWO independently augmented views of every
    sample, concatenate them to get a [2B, 1024] input, embed to [2B, 256],
    and compute the contrastive loss over all 2B embeddings.
    Doubling the views maximises the number of positive pairs per batch.
    """
    model.train()
    total_loss, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Two independently augmented views
        x1 = augment_vec(x, noise_std=noise_std)
        x2 = augment_vec(x, noise_std=noise_std)
        x_both = torch.cat([x1, x2], dim=0)   # [2B, 1024]
        y_both = torch.cat([y,  y],  dim=0)   # [2B]

        optimizer.zero_grad()
        emb  = model(x_both)                  # [2B, 256]
        loss = supcon_loss(emb, y_both)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total      += x.size(0)
    return total_loss / total


# ─────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────
def save_checkpoint(path, model, le, epoch, prototypes: np.ndarray):
    """
    Save model weights, label encoder, and per-individual prototype embeddings.

    prototypes : np.ndarray [num_classes, emb_dim]
        Row i = L2-normalised mean embedding of individual le.classes_[i],
        computed over the full training set after the final epoch.
        Used by --test mode for cosine nearest-neighbour prediction.
    """
    torch.save({
        'epoch':       epoch,
        'model_state': model.state_dict(),
        'le_classes':  le.classes_,
        'emb_dim':     model.emb_dim,
        'prototypes':  prototypes,
    }, path)


def load_checkpoint(path, device=DEVICE):
    ck = torch.load(path, map_location=device, weights_only=False)

    emb_dim = ck['emb_dim']
    model   = BirdCNN(emb_dim=emb_dim).to(device)
    model.load_state_dict(ck['model_state'])
    model.eval()

    le          = LabelEncoder()
    le.classes_ = ck['le_classes']
    prototypes  = ck['prototypes']   # [num_classes, emb_dim]

    print(f"Loaded checkpoint: epoch={ck['epoch']}, "
          f"emb_dim={emb_dim}, num_classes={len(le.classes_)}, "
          f"classes={list(le.classes_)}")
    return model, le, prototypes


# ─────────────────────────────────────────────────────────────
# Batched inference
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_vecs(model, vecs_np: np.ndarray, device, batch_size=256) -> np.ndarray:
    """
    Push raw 1024-d feature vectors through the CNN.
    Returns 256-d normalised embeddings [N, 256].
    """
    model.eval()
    out = []
    for start in range(0, len(vecs_np), batch_size):
        chunk = torch.from_numpy(
            vecs_np[start: start + batch_size]
        ).float().to(device)
        out.append(model(chunk).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def compute_prototypes(model, dataset: EmbeddingDataset,
                       device, batch_size: int = 256) -> np.ndarray:
    """
    Compute per-class prototype embeddings over the full training set.

    For each individual, all segment embeddings are mean-pooled and
    re-normalised to produce one L2-unit prototype vector.

    Returns np.ndarray [num_classes, emb_dim].
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=NUM_WORKERS)
    all_embs, all_labels = [], []
    for x, y in loader:
        all_embs.append(model(x.to(device)).cpu().numpy())
        all_labels.append(y.numpy())

    all_embs   = np.concatenate(all_embs,   axis=0)   # [N, D]
    all_labels = np.concatenate(all_labels, axis=0)   # [N]
    num_classes = len(dataset.le.classes_)
    D           = all_embs.shape[1]

    prototypes = np.zeros((num_classes, D), dtype=np.float32)
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            mean_emb      = all_embs[mask].mean(axis=0)
            prototypes[c] = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

    return prototypes


# ─────────────────────────────────────────────────────────────
# Mode: train
# ─────────────────────────────────────────────────────────────
def run_train(args):
    # ── Load data ────────────────────────────────────────────
    df = load_embeddings_csv(args.embeddings_csv, args.labels_csv)
    if df['individual_id'].eq('').any():
        raise ValueError(
            "Training requires individual_id for every row. "
            "Add an individual_id column to your embeddings CSV "
            "or provide --labels_csv."
        )

    # Class balance report
    counts = df['individual_id'].value_counts()
    print("\nSamples per individual (training data):")
    print(counts.to_string())
    print()

    le          = LabelEncoder()
    num_classes = len(le.fit(df['individual_id'].tolist()).classes_)
    print(f"Individuals: {num_classes}  |  Total segments: {len(df)}")

    train_ds = EmbeddingDataset(df, label_encoder=le)

    # Balanced sampler: guarantees ≥n_samples per class per batch
    try:
        sampler  = BalancedBatchSampler(
            train_ds.labels,
            n_classes=min(num_classes, args.supcon_n_classes),
            n_samples=args.supcon_n_samples,
        )
        train_dl = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=NUM_WORKERS)
        print(f"SupCon batch: {sampler.n_classes} classes × "
              f"{sampler.n_samples} samples = {sampler.batch_size} per batch")
    except ValueError as e:
        print(f"[WARNING] BalancedBatchSampler: {e}")
        print("[WARNING] Falling back to standard DataLoader — "
              "some batches may lack positive pairs.")
        train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    # ── Model ────────────────────────────────────────────────
    model     = BirdCNN(emb_dim=EMBEDDING_DIM).to(DEVICE)
    supcon    = SupConLoss(temperature=args.supcon_temp).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNN parameters: {param_count:,}\n")

    # ── Training loop ────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, supcon, train_dl, optimizer, DEVICE)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{args.epochs}  supcon_loss={tr_loss:.4f}")

    # ── Compute prototypes over full training set ─────────────
    print("\nComputing per-individual prototype embeddings …")
    prototypes = compute_prototypes(model, train_ds, DEVICE)
    print(f"Prototypes shape: {prototypes.shape}  (one L2-unit vector per individual)")

    # Save final checkpoint
    final_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    save_checkpoint(final_path, model, le, args.epochs, prototypes)
    print(f"Final model saved: {final_path}")


# ─────────────────────────────────────────────────────────────
# Mode: test
# ─────────────────────────────────────────────────────────────
def run_test(args):
    df = load_embeddings_csv(args.embeddings_csv)
    model, le, prototypes = load_checkpoint(args.checkpoint, device=DEVICE)

    # Embed all segments
    vecs = np.stack(df['embedding_vec'].values)       # [N, 1024]
    embs = embed_vecs(model, vecs, DEVICE)             # [N, 256]

    # Cosine similarity to prototypes → predicted class index per segment
    # prototypes : [num_classes, 256], already L2-normalised
    cos_sim   = embs @ prototypes.T                    # [N, num_classes]
    seg_preds = cos_sim.argmax(axis=1)                 # [N]

    # Aggregate per file: majority vote across segments
    df['_pred_idx'] = seg_preds
    results = []
    for fp, grp in df.groupby('file_path', sort=False):
        votes    = grp['_pred_idx'].values
        pred_idx = int(np.bincount(votes).argmax())
        pred_lbl = (str(le.classes_[pred_idx])
                    if pred_idx < len(le.classes_) else f"unknown_{pred_idx}")
        confidence = float((votes == pred_idx).mean())
        results.append({'file_path': fp,
                        'predicted_individual': pred_lbl,
                        'confidence': f"{confidence:.2f}"})

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_pred_csv, index=False)
    print(f"\nPredictions saved → {args.out_pred_csv}")
    print(out_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# Mode: embedding
# ─────────────────────────────────────────────────────────────
def run_embedding(args):
    df = load_embeddings_csv(args.embeddings_csv)
    model, _, _ = load_checkpoint(args.checkpoint, device=DEVICE)

    # Embed all segments
    vecs = np.stack(df['embedding_vec'].values)        # [N, 1024]
    embs = embed_vecs(model, vecs, DEVICE)             # [N, 256]

    # Aggregate per file: mean pool over segments, then re-normalise
    df['_seg_idx'] = range(len(df))
    file_embs  = []
    file_paths = []
    for fp, grp in df.groupby('file_path', sort=False):
        idxs     = grp['_seg_idx'].values
        mean_emb = embs[idxs].mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        file_embs.append(mean_emb)
        file_paths.append(fp)

    file_embs_arr = np.stack(file_embs)                # [F, 256]
    np.save(args.out_emb_npy, file_embs_arr)

    # Save file_path → embedding row index mapping
    meta_csv = os.path.splitext(args.out_emb_npy)[0] + "_meta.csv"
    pd.DataFrame({'file_path': file_paths,
                  'emb_row': range(len(file_paths))}).to_csv(meta_csv, index=False)

    print(f"\nEmbeddings saved → {args.out_emb_npy}  shape={file_embs_arr.shape}")
    print(f"Metadata  saved → {meta_csv}")
    print(f"Each row is a 256-d L2-normalised identity embedding for one file.")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="BirdNet 1024-d feature CNN pipeline for individual recognition",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--mode', required=True,
        choices=['train', 'test', 'embedding'],
        help=(
            "train    : Train CNN on labelled BirdNet embeddings\n"
            "test     : Predict individual ID for unlabelled files\n"
            "embedding: Extract 256-d identity vectors"
        )
    )

    # ── Data ──────────────────────────────────────────────────
    parser.add_argument(
        '--embeddings_csv', type=str, required=True,
        help="Path to BirdNet feature CSV (file_path, start, end, embedding)"
    )
    parser.add_argument(
        '--labels_csv', type=str, default=None,
        help="(Optional) Separate CSV (file_path, individual_id) for training labels.\n"
             "Not needed if individual_id column is already in --embeddings_csv."
    )
    parser.add_argument(
        '--val_embeddings_csv', type=str, default=None,
        help="Validation set embeddings CSV (reserved for future use)"
    )
    parser.add_argument(
        '--val_labels_csv', type=str, default=None,
        help="Validation set labels CSV (reserved for future use)"
    )

    # ── Training ──────────────────────────────────────────────
    parser.add_argument('--epochs',           type=int,   default=EPOCHS)
    parser.add_argument('--batch_size',       type=int,   default=64,
                        help="Fallback batch size if BalancedBatchSampler cannot be used")
    parser.add_argument('--lr',               type=float, default=LR)
    parser.add_argument('--supcon_temp',      type=float, default=SUPCON_TEMP,
                        help=f"SupCon temperature τ (default {SUPCON_TEMP}). "
                             "Lower = harder negatives / tighter clusters.")
    parser.add_argument('--supcon_n_classes', type=int,   default=SUPCON_N_CLASSES,
                        help=f"Individuals per balanced batch (default {SUPCON_N_CLASSES})")
    parser.add_argument('--supcon_n_samples', type=int,   default=SUPCON_N_SAMPLES,
                        help=f"Segments per individual per batch (default {SUPCON_N_SAMPLES})")
    parser.add_argument('--checkpoint_dir',   type=str,   default=CHECKPOINT_DIR)

    # ── Inference ─────────────────────────────────────────────
    parser.add_argument('--checkpoint',    type=str, default=None,
                        help="Path to trained .pt checkpoint (test / embedding modes)")
    parser.add_argument('--out_pred_csv',  type=str, default='predictions.csv')
    parser.add_argument('--out_emb_npy',   type=str, default='embeddings.npy')

    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Mode   : {args.mode}\n")

    if args.mode == 'train':
        run_train(args)

    elif args.mode == 'test':
        if not args.checkpoint:
            parser.error("--checkpoint is required for test mode.")
        run_test(args)

    elif args.mode == 'embedding':
        if not args.checkpoint:
            parser.error("--checkpoint is required for embedding mode.")
        run_embedding(args)


if __name__ == '__main__':
    main()
