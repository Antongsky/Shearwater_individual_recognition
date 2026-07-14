This Python script (ana.py) is used to do clustring of numpy arrays output from the previous CNN embedding head.

CNN embedding head outputs np array of 256d embeddings from Manx Shearwater call clips (shape: [n, 256], for n is the generated clip number for a sample). This script takes the np array (.npy file) as input, and predicts the optimal cluster number as predictioon individual number.

ana_multifile.py -> Can batch process embeddings of multiple file results at one time, and give results. Input shape: [m,n,256], where m is sample mnumber, n is the generated clip number for a sample.
Individual number estimation formula: use the cluster number x corresponding to the global maximum of the silhouette score as estimation.

ana_multifile_v3.py -> Modified from ana_multifile.py to optimize the individual number estimation formula: If finding local maximum(s) that has height >= 0.9 of the global maximum, use the x for the first local maximum as the prediction, rather than x of the global maximum.

How to choose:
For one audio recording: ana.py
For batch analysis of multiple samples: ana_multifile.py and ana_multifile_v3.py
*ana_multifile_v3.py may work better for recordings containing smaller number of individuals (<20).
