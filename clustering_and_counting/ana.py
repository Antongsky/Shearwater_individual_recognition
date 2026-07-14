import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.signal import savgol_filter



def agglomerative_compute(embeddings):
    pca = PCA(n_components = 2)   # Change n_components in parameter test
    embeddings = pca.fit_transform(embeddings)
    cluster_vec = []
    score_vec = []
    for n_clusters in range (2, len(embeddings)):
        cluster_agg = AgglomerativeClustering(n_clusters, metric = "cosine", linkage = "average")
        labels = cluster_agg.fit_predict(embeddings)
        score = silhouette_score(embeddings,labels , metric = "cosine")
        cluster_vec.append(n_clusters)
        score_vec.append(score)
    return cluster_vec, score_vec



path = ""                   #Set path for the embrdding file
embeddings = np.load(path)
    
cluster, score = agglomerative_compute(embeddings)
score = np.asarray(score)
score_s = savgol_filter(score, window_length = len(cluster) // 4, polyorder = 3) 
cluster = np.asarray(cluster)
threshold = np.max(score_s) * 0.9
print("point prediction:", cluster[np.where(score_s == np.max(score_s))[0]][0])
print("Range Prediction:", cluster[np.where(score_s >= threshold)[0]][0], cluster[np.where(score_s >= threshold)[0]][-1])
    
plt.plot(cluster, score_s)
plt.show()






