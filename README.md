## 1 About this depository
This repository is for identifying Manx shearwater individual by their calls, and counting individual numbers in audio recordings, using machine learning. 

This repository is created by Zhanyi Lin (Imperial College London) in July 2026, for the Imperial College London Master project "Predicting Manx shearwater individual number in acoustic recordings using machine learning". The project is funded by Imperial College London, and field research grant is offered by Lundy Field Society.

## 2 Content

./join_small_files			-- join clips of individual calls to longer recordings containing multiple individuals, for artificially creating samples for some usage. 

./generate_birdnet_clip		-- Generate short clips using birdnet recognition.

./1D-CNN model				-- Train or test the model of indiviudal recognition. Or, output 256-d embeddings for open-set individual number counting.

./clustering_and_counting	-- Agglomerative clustering followed by indiviudal estimation. 

./data_analysis				-- Data analysis and plot using R.

Readmes are included in each folder for more detailed descriptions.

## 3 Usage

-- Data preparation: Audio files containing Manx shearwater calls. Seperate training and test sets.

-- Clip detection: using ./generate_birdnet_clip to generate clips containing their calls.

-- Embedding extration using BirdNet GUI (not in this dipository). Output: 1024-d embeddings in a CSV file.

-- Model training: ./1D-CNN model - Train mode.

-- a Model testing: ./1D-CNN model - Test mode.

-- b Embedding extration: ./1D-CNN model - Embed mode. Input: 1024-d embeddings extracted from files containing multiple individual calls -> clip detection -> embedding extraction. Output: npy file containing 256-d embeddings.

-- b Indiviudal number prediction: ./clustering_and_counting.

## 4 Data

Data for this research can be found here: https://doi.org/10.5281/zenodo.21297859
which includes model checkpoint pt files, training set and test set clip embeddings, test set log file, etc.
