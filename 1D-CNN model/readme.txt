Using the pipeline

1. Naming
training set files should be correctly named. Individual ID should be put after a "_" in the end of the  file name. E.g., Lundy_12345_01_ix.WAV. The indiviudal ID is "ix" and can be successfully extracted by add_id.py. If using a different naming allocatoin, then change add_id.py.

2. Birdnet feature embedding extraction
Extract 1024d feature embeddings using birdnet gui, and export as csv files

3. Model training
-> First modify the training set feature embedding csv to add a id column (add_id.py)
-> Check wheather each individual has enough samples; delete individuals lack of samples and redundant samples (pre_run_check.R)
-> Find the correct model version: V2 - Training epoch = n; V3 - first train and validate using 85% and 15% of data, finding the best epoch E. Then the second round train on full training set in epoch E.
-> Then run 
[python birdnet_cnn_pipeline[version].py --mode train --embeddings_csv feature_embeddings.csv --epochs n --batch_size 32 --lr 0.001 --checkpoint_dir ./checkpoints]

### Other parameters such as SupCon temp, dropout rate, can be changed in the Python script py file.


4. Model prediction
[python birdnet_cnn_pipeline[version].py --mode test --embeddings_csv ./test_feature_embeddings.csv --checkpoint ./checkpoints/final_model.pt --out_pred_csv ./predictions.csv]

5. Model embedding extraction
[python birdnet_cnn_pipeline[version].py --mode embedding --embeddings_csv ./test_feature_embeddings.csv --checkpoint ./checkpoints/final_model.pt --out_emb_npy ./embeddings_256d.npy]

