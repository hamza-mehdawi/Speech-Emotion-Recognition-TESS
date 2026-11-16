"""
Extract MFCC features from the same TESS audio files used in the CNN notebook.

This file is separate from the CNN model notebook.
It does not train a model. It only creates MFCC feature numbers and saves them.

Outputs:
- outputs/mfcc_features/mfcc_features_all_files.csv
- outputs/mfcc_features/mfcc_features_all_files.npz
- outputs/mfcc_features/mfcc_feature_summary.csv
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Paths and audio settings
# ---------------------------------------------------------------------

# This is the same dataset path used in the CNN notebook.
DATASET_PATH = Path(
    r"D:\Deep Lerning\project\new projects\Toronto emotional speech set (TESS)\TESS"
)

# MFCC outputs are saved in their own folder.
OUTPUT_DIR = Path("outputs") / "mfcc_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# These match the CNN preprocessing settings.
SAMPLE_RATE = 22050
DURATION_SECONDS = 3.0

# Number of MFCC coefficients.
# 13 is a common beginner-friendly choice for speech features.
N_MFCC = 13

# Same FFT settings used in the CNN spectrogram code.
N_FFT = 2048
HOP_LENGTH = 512


# ---------------------------------------------------------------------
# 2. Dataset scanning
# ---------------------------------------------------------------------

def scan_tess_dataset(dataset_path):
    """Find all WAV files and create a table with useful label information."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder was not found: {dataset_path}")

    rows = []
    wav_files = sorted(dataset_path.rglob("*.wav"))

    for file_path in wav_files:
        # TESS files usually look like:
        # OAF_angry/OAF_back_angry.wav
        folder_name = file_path.parent.name
        file_stem = file_path.stem

        parts = file_stem.split("_")
        speaker = parts[0] if len(parts) > 0 else "unknown"
        word = parts[1] if len(parts) > 1 else "unknown"
        emotion = folder_name.split("_")[-1].lower()

        # Keep the same class name style used in the project.
        if emotion == "ps":
            emotion = "pleasant_surprise"

        rows.append(
            {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "speaker": speaker,
                "word": word,
                "emotion": emotion,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. Audio loading
# ---------------------------------------------------------------------

def load_audio_file(file_path, sample_rate=SAMPLE_RATE, duration_seconds=DURATION_SECONDS):
    """Load audio using the same simple preprocessing idea as the CNN notebook."""
    # Load as mono and resample to 22050 Hz.
    audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)

    # Make every audio file exactly 3 seconds.
    target_length = int(sample_rate * duration_seconds)

    if len(audio) < target_length:
        # If audio is short, add silence at the end.
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        # If audio is long, keep only the first 3 seconds.
        audio = audio[:target_length]

    # Normalize volume to a similar range.
    max_value = np.max(np.abs(audio))
    if max_value > 0:
        audio = audio / max_value

    return audio.astype(np.float32)


# ---------------------------------------------------------------------
# 4. MFCC extraction
# ---------------------------------------------------------------------

def extract_mfcc(file_path):
    """Extract the raw MFCC matrix for one audio file."""
    audio = load_audio_file(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    return mfcc.astype(np.float32)


def summarize_mfcc(mfcc):
    """
    Convert an MFCC matrix into one fixed-size feature vector.

    MFCC shape is:
    - rows: MFCC coefficients
    - columns: time frames

    We use mean and standard deviation for each coefficient.
    This gives 13 means + 13 standard deviations = 26 numbers per file.
    """
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    return np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)


def create_feature_column_names(n_mfcc=N_MFCC):
    """Create clear column names for the MFCC feature table."""
    mean_columns = [f"mfcc_{index:02d}_mean" for index in range(1, n_mfcc + 1)]
    std_columns = [f"mfcc_{index:02d}_std" for index in range(1, n_mfcc + 1)]
    return mean_columns + std_columns


def extract_mfcc_features_for_dataset(metadata_df):
    """Extract MFCC feature numbers for every audio file."""
    feature_rows = []
    feature_vectors = []

    for row_number, row in metadata_df.iterrows():
        if row_number % 100 == 0:
            print(f"Extracting MFCC features: {row_number}/{len(metadata_df)}")

        mfcc = extract_mfcc(row["file_path"])
        feature_vector = summarize_mfcc(mfcc)
        feature_vectors.append(feature_vector)

        output_row = {
            "file_path": row["file_path"],
            "file_name": row["file_name"],
            "speaker": row["speaker"],
            "word": row["word"],
            "emotion": row["emotion"],
        }

        for column_name, value in zip(create_feature_column_names(), feature_vector):
            output_row[column_name] = value

        feature_rows.append(output_row)

    features_df = pd.DataFrame(feature_rows)
    X_mfcc = np.vstack(feature_vectors).astype(np.float32)
    y_emotion = metadata_df["emotion"].to_numpy()

    return features_df, X_mfcc, y_emotion


# ---------------------------------------------------------------------
# 5. Run the extraction
# ---------------------------------------------------------------------

def main():
    print("Dataset path:", DATASET_PATH)
    print("Output folder:", OUTPUT_DIR.resolve())

    metadata_df = scan_tess_dataset(DATASET_PATH)
    print("Total WAV files found:", len(metadata_df))
    print("Emotion classes:", sorted(metadata_df["emotion"].unique()))

    features_df, X_mfcc, y_emotion = extract_mfcc_features_for_dataset(metadata_df)

    csv_path = OUTPUT_DIR / "mfcc_features_all_files.csv"
    npz_path = OUTPUT_DIR / "mfcc_features_all_files.npz"
    summary_path = OUTPUT_DIR / "mfcc_feature_summary.csv"

    features_df.to_csv(csv_path, index=False)

    np.savez_compressed(
        npz_path,
        X_mfcc=X_mfcc,
        y_emotion=y_emotion,
        file_paths=metadata_df["file_path"].to_numpy(),
        feature_names=np.array(create_feature_column_names()),
    )

    # This summary shows average MFCC feature values per emotion.
    feature_columns = create_feature_column_names()
    summary_df = features_df.groupby("emotion")[feature_columns].mean()
    summary_df.to_csv(summary_path)

    print("\nDone.")
    print("MFCC feature matrix shape:", X_mfcc.shape)
    print("Saved CSV:", csv_path)
    print("Saved NPZ:", npz_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
