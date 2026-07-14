import pandas as pd
import re

# Load the CSV
df = pd.read_csv("C://Users//26739//Desktop//CNN_embedding_head//bn_embedding//embeddings.csv")
output_path = "C://Users//26739//Desktop//CNN_embedding_head//bn_embedding//embeddings_id.csv"

# Extract individual_id from file_path:
def extract_individual_id(file_path):
    match = re.search(r'\$(.+?)\$', file_path)
    if match:
        between_dollars = match.group(1)  
        return between_dollars.split('_')[-1]  
    return None

df['individual_id'] = df['file_path'].apply(extract_individual_id)

# Save output

df.to_csv(output_path, index=False)

print("Done! Sample of individual_ids extracted:")
print(df[['file_path', 'individual_id']].head(10).to_string())
