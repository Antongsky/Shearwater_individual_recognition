import os
import random
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt




def join_wav(input_dir, output_dir, a, b, n, x): # Select randomly [a,b] files. Generare n long files. x is the identifier for the batch

    os.makedirs(output_dir, exist_ok=True)

    # Get all wav files
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]

    for i in range(n):
        k = random.randint(a, b)

        # Randomly select files
        selected_files = random.sample(wav_files, min(k, len(wav_files)))

        joined_audio = []
        total_duration = 0
        used_ids = []

        for file in selected_files:
            filepath = os.path.join(input_dir, file)

            # Extract individual ID
            individual_id = file.split("_")[0]
            used_ids.append(individual_id)

            # Load and resample to 48kHz
            audio, sr = librosa.load(filepath, sr=48000)


            duration = len(audio) / 48000

            # Ensure total length < 20 minutes
            if total_duration + duration > 1200:
                break

            joined_audio.append(audio)
            total_duration += duration

        if len(joined_audio) == 0:
            continue

        # Concatenate
        final_audio = np.concatenate(joined_audio)

        # Count unique individuals
        unique_ids = set(used_ids)
        individual_count = len(unique_ids)

        # Generate random 3-digit file index
        file_index = str(i + 1).zfill(3)

        output_filename = f"join_{x}{file_index}_{individual_count}.wav"
        output_path = os.path.join(output_dir, output_filename)

        # Save
        sf.write(output_path, final_audio, 48000)

        print(f"Saved: {output_filename} | Individuals: {individual_count} | Duration: {round(total_duration,2)} sec")


if __name__ == "__main__":
    # Example usage
    join_wav(input_dir="D://OneDrive - Imperial College London//project//project_test//net_recordings//Sun et al 2022//", 
             output_dir="D://OneDrive - Imperial College London//project//project_test//net_recordings//res//",
        a=15, b=35, n=6, x = "D")