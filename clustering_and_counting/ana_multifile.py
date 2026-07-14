import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os
#------------------------------------------------------------------------------


# Configuration
emb_path  = "C://Users//testSampCtl10_rep1.npy" #NPY file path from model output using EMBED mode
meta_path = "C://Users//testSampCtl10_rep1_meta.csv"   # Meta file also from the core model
out_path   = "C://Users//estSampCtl10_res.csv"   #Output csv dir and name e.g., ".../abc.csv"
n_components = 2
PLOT      = False   # set False to skip plots, otherwise show diagnostic plots


# Functions--------------------------------------------------------------------
def extract_sample_name(file_path: str):  #When doing clip generation, clips are named like "Clip$file_name$01.wav"
    parts = os.path.basename(file_path).split("$")
    if len(parts) < 3:
        raise ValueError(f"Could not parse sample name from: {file_path}")
    return parts[1]


def agglomerative_compute(embeddings: np.ndarray, n_components):

    pca = PCA(n_components= n_components)
    emb_pca = pca.fit_transform(embeddings)

    cluster_vec, score_vec = [], []
    for k in range(2, len(embeddings)):         
        model  = AgglomerativeClustering(k, metric="cosine", linkage="average")
        labels = model.fit_predict(emb_pca)
        score  = silhouette_score(emb_pca, labels, metric="cosine")
        cluster_vec.append(k)
        score_vec.append(score)
    return cluster_vec, score_vec


def predict_from_scores(cluster_vec, score_vec, window_frac=0.2, polyorder=3):

 #Smooth the silhouette curve with a Savitzky-Golay filter and return point prediction (peak) and range prediction (90% of peak).
    cluster = np.asarray(cluster_vec)
    score   = np.asarray(score_vec)
    wlen = max(polyorder + 2, int(len(cluster) * window_frac) | 1)
    if wlen > len(cluster):
        wlen = len(cluster) if len(cluster) % 2 == 1 else len(cluster) - 1
    wlen = max(wlen, polyorder + 2)

    score_s   = savgol_filter(score, window_length=wlen, polyorder=polyorder)
    threshold = np.max(score_s) * 0.9  #Can also adjust this threshold

    point_pred = int(cluster[np.argmax(score_s)])
    range_low  = int(cluster[np.where(score_s >= threshold)[0][0]])
    range_high = int(cluster[np.where(score_s >= threshold)[0][-1]])

    return point_pred, range_low, range_high, cluster, score_s


# Load data -------------------------------------------------------------------
print("Loading embeddings and metadata …")
embeddings = np.load(emb_path)
meta       = pd.read_csv(meta_path, sep=",")  

# Extract sample name for every row in meta
meta["sample_name"] = meta["file_path"].apply(extract_sample_name)

print(f"  Embeddings shape : {embeddings.shape}")
print(f"  Metadata rows    : {len(meta)}")
print(f"  Unique samples   : {meta['sample_name'].nunique()}\n")

# Sample analysis -------------------------------------------------------------
results = []

for sample_name, group in meta.groupby("sample_name"):
    row_indices = group["emb_row"].tolist()
    sub_emb = embeddings[row_indices]          # shape: [n_clip, 256]
    print(f"[{sample_name}]  segments: {len(sub_emb)}")

    cluster_vec, score_vec = agglomerative_compute(sub_emb, n_components = n_components)
    point_pred, range_low, range_high, cluster, score_s = predict_from_scores(cluster_vec, score_vec)
    print(f"  Point prediction : {point_pred}")
    print(f"  Range prediction : {range_low} – {range_high}\n")

    results.append({"sample_name" : sample_name, "n_segments" : len(sub_emb), "point_pred" : point_pred, "range_low" : range_low,"range_high" : range_high,})

    if PLOT:
        plt.figure(figsize=(7, 3))
        plt.plot(cluster, score_s, linewidth=1.5)
        plt.axvline(point_pred, color="red",    linestyle="--", label=f"Point pred = {point_pred}")
        plt.axvspan(range_low, range_high, alpha=0.15, color="orange", label=f"Range {range_low}–{range_high}")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette score (smoothed)")
        plt.title(f"{sample_name}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

# Save the output results to out_path------------------------------------------
out_df = pd.DataFrame(results)
out_df.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
print(out_df.to_string(index=False))
