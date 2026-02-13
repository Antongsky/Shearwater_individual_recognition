import os
import librosa
import soundfile as sf
import birdnet 
import pandas as pd

def time_to_seconds(t):
    """
    Convert time string 'HH:MM:SS.xx' to seconds (float).
    """
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def birdnet_clip_audio(audio_dir, save_dir, threshold = 0.8):
    """
    Detect Manx Shearwater using BirdNET and save 3-second clips
    above confidence threshold.

    Parameters
    ----------
    audio_dir : str
        Directory containing wav/WAV files.
    save_dir : str
        Directory to save clipped audio.
    threshold : float
        Confidence threshold (0-1).
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    clip_counter = 0

    for file in os.listdir(audio_dir):
        file_name = file.split(".")[0]
        if not file.lower().endswith(".wav"):
            continue

        file_path = os.path.join(audio_dir, file)
        print(f"Processing {file}...")

        # Run BirdNET prediction
        model = birdnet.load("acoustic", "2.4", "tf")
        predictions = model.predict(file_path)
        predictions.to_csv(f"{save_dir}{file_name}.csv")

        # Read the csv and do clip
        df = pd.read_csv(f"{save_dir}{file_name}.csv")
        clip_id = 1
        
        audio, sr = librosa.load(file_path, sr=None)
        
        filtered = df[
            (df["species_name"].str.contains("Manx Shearwater", case=False)) &
            (df["confidence"] >= threshold)
        ]

        for _, row in filtered.iterrows():

            start_sec = time_to_seconds(row["start_time"])
            #end_sec = time_to_seconds(row["end_time"])

            start_sample = int(start_sec * sr)
            end_sample = start_sample + 3 * sr

            clip = audio[start_sample:end_sample]
            clip_name = f"Clip${file_name}${clip_id}.wav"
            save_path = os.path.join(save_dir, clip_name)

            sf.write(save_path, clip, sr)
            clip_id += 1
            clip_counter += 1
        print(f"Saved {clip_counter} clips for file {file}.")
    
    print(f"Finished. Saved {clip_counter} clips.")


if __name__ == "__main__":
    birdnet_clip_audio("C://Users//26739//Desktop//11//", "C://Users//26739//Desktop//12//")