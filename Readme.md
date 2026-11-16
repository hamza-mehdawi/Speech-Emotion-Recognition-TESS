# TESS Speech Emotion Recognition (SER)

A deep learning research and comparative benchmarking project for **Speech Emotion Recognition** using the **Toronto Emotional Speech Set (TESS)**. This repository implements, tunes, and evaluates multiple deep neural network architectures—including **2D Convolutional Neural Networks (CNN)**, **Recurrent Neural Networks (RNN)**, and **Vision Transformers (ViT)**—alongside a noise-robustness experiment, a classical **MFCC feature extraction** pipeline, and project documentation.

## 📋 Table of Contents

* [Project Overview](#-project-overview)
* [Dataset](#-dataset)
* [Project Structure](#-project-structure)
* [Audio Preprocessing & Feature Engineering](#-audio-preprocessing--feature-engineering)
* [Model Architectures](#-model-architectures)
* [Installation & Setup](#-installation--setup)
* [Usage & Workflow](#-usage--workflow)
* [Evaluation & Research Comparison](#-evaluation--research-comparison)

## 🎯 Project Overview

Recognizing human emotion from speech is a core challenge in affective computing and human-computer interaction. In this project, raw `.wav` audio files are transformed into 2D time-frequency representations (**Log-Mel Spectrograms**) and evaluated across distinct deep learning paradigms:

1. **Baseline 2D CNN:** Treats Log-Mel spectrograms as single-channel images to capture local time-frequency patterns.
2. **Noise-Robust CNN (5% Gaussian Noise):** Evaluates model resilience when trained and tested on audio injected with low-level Gaussian noise ($\sigma = 0.05$).
3. **Recurrent Neural Network (RNN):** Treats the spectrogram as a sequential time-series across frequency bins.
4. **Vision Transformer (ViT):** Splits spectrograms into non-overlapping $8 \times 8$ patches and applies multi-head self-attention to capture global time-frequency dependencies.
5. **Research Comparison Suite:** Automatically aggregates accuracy, F1-score, ROC-AUC, parameter counts, inference latency, and Pareto frontiers across all trained models.

## 🗂️ Dataset

The project uses the **Toronto Emotional Speech Set (TESS)**, which consists of spoken target words recorded by two actresses (aged 26 and 64—labeled `YAF` and `OAF`) across **7 emotion classes**:

* `angry`
* `disgust`
* `fear`
* `happy`
* `neutral`
* `pleasant_surprise` (`ps`)
* `sad`

> **Note:** Update `DATASET_PATH` in the configuration cell of the notebooks/scripts to point to your local TESS directory before running.

## 📁 Project Structure

```text
├── code/                                                           # Source notebooks & feature extraction scripts
│   ├── TESS_CNN_Audio_Emotion_Recognition.ipynb                    # Baseline 2D CNN pipeline
│   ├── TESS_CNN_Low_Gaussian_Noise_Audio_Emotion_Recognition.ipynb # CNN trained with 5% Gaussian noise
│   ├── TESS_RNN_Audio_Emotion_Recognition.ipynb                    # Sequence-based SimpleRNN pipeline
│   ├── TESS_ViT_Audio_Emotion_Recognition.ipynb                    # Patch-based Vision Transformer pipeline
│   ├── TESS_Deep_Models_Research_Comparison.ipynb                  # Automated Pareto & efficiency comparison
│   └── extract_mfcc_features_tess.py                               # Standalone MFCC tabular feature extractor
├── outputs/                                                        # Generated models, metrics & comparison artifacts
│   └── deep_model_comparison/                                      # Cross-model efficiency tables & Pareto plots
├── Speech Emotion Recognition Al najah.pdf                         # Project presentation / academic report
├── Speech Emotion Recognition Application.pdf                      # Application documentation & overview
└── tess_cnn_audio_pipeline.html                                    # Interactive HTML visualization of the CNN pipeline
```

## 🎛️ Audio Preprocessing & Feature Engineering

All deep learning pipelines share a standardized preprocessing workflow (`SAMPLE_RATE = 22050 Hz`, `DURATION = 3.0s`):

1. **Standardization:** Audio is loaded in mono, resampled to $22,050\text{ Hz}$, and padded or truncated to a fixed length of $66,150$ samples ($3.0\text{ seconds}$).
2. **Amplitude Normalization:** Waveforms are peak-normalized to $[-1.0, 1.0]$.
3. **Log-Mel Spectrograms:** Computed using `n_mels=128`, `n_fft=2048`, and `hop_length=512`, converted to decibels, and min-max scaled to $[0, 1]$, resulting in tensors of shape `(128, 130, 1)`.
4. **Feature Caching:** Extracted spectrograms are cached in `outputs/features_cache.npz` when running the pipelines to accelerate subsequent runs across CNN, RNN, and ViT models.
5. **Classical MFCC Extraction:** `code/extract_mfcc_features_tess.py` computes $13$ Mel-Frequency Cepstral Coefficients (mean and standard deviation across time frames, yielding $26$ features per audio file) for traditional machine learning or statistical analysis.

## 🧠 Model Architectures

| Model | Input Representation | Core Architecture | Key Features |
| :--- | :--- | :--- | :--- |
| **CNN** | `(128, 130, 1)` Image | 3× `Conv2D` + `BatchNorm` + `MaxPool2D` + `Dropout` | Local spatial-spectral feature extraction; `128`-dim embedding layer |
| **CNN (Noisy)** | `(128, 130, 1)` + $\mathcal{N}(0, 0.05^2)$ | Same as Baseline CNN | Tests generalization under additive white Gaussian noise |
| **RNN** | `(130, 128)` Sequence | `SimpleRNN` + `BatchNorm` + `Dropout` | Processes temporal progression across Mel frequency bands |
| **ViT** | `(128, 130, 1)` Patches | Custom `PatchExtractor` ($8\times8$) + `PatchEncoder` + `MultiHeadAttention` | Global self-attention across time-frequency spectrogram patches; GPU-accelerated |

Each model includes:

* Stratified **70% Train / 15% Validation / 15% Test** splitting (`RANDOM_STATE = 42`).
* Hyperparameter grid search (`learning_rate`, `dropout_rate`, `batch_size`, `model_size`) with `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`.
* Live microphone recording (`sounddevice`) and single-file `.wav` inference utilities.

## ⚙️ Installation & Setup

Ensure you have **Python 3.9+** installed, then install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn librosa soundfile tensorflow sounddevice jupyter
```

*(Optional)* For GPU acceleration with the Vision Transformer (`FORCE_GPU_FOR_VIT = True`), ensure CUDA and cuDNN are properly configured for your TensorFlow installation.

## 🚀 Usage & Workflow

### 1. Train & Evaluate Individual Models

Navigate to the `code/` folder and run any of the model Jupyter notebooks independently. Each notebook scans the dataset, generates exploratory audio statistics, trains hyperparameter trials, saves the best `.keras` model, and outputs evaluation plots:

* `code/TESS_CNN_Audio_Emotion_Recognition.ipynb`
* `code/TESS_CNN_Low_Gaussian_Noise_Audio_Emotion_Recognition.ipynb`
* `code/TESS_RNN_Audio_Emotion_Recognition.ipynb`
* `code/TESS_ViT_Audio_Emotion_Recognition.ipynb`

### 2. Run the Cross-Model Research Comparison

Once the individual models have generated their `model_performance_complexity.csv` reports, run the comparison notebook to generate efficiency metrics, Markdown tables, and Pareto frontier charts inside `outputs/deep_model_comparison/`:

* `code/TESS_Deep_Models_Research_Comparison.ipynb`

*(Optional)* Inside `TESS_Deep_Models_Research_Comparison.ipynb`, you can configure `CLOUD_GPU_COST_PER_HOUR` and `MANUAL_FLOPS_BILLIONS` to compute cost-per-accuracy and FLOP-efficiency metrics.

### 3. Extract Tabular MFCC Features

To generate standalone CSV and compressed NumPy datasets of MFCC features:

```bash
python code/extract_mfcc_features_tess.py
```

## 📊 Evaluation & Research Comparison

When executed, each model generates diagnostic artifacts in `outputs/` (or its respective `outputs/<model_family>/` subdirectory):

* **Classification Metrics:** `classification_report.txt`, `test_metrics.csv`, `confusion_matrix.png`
* **ROC & AUC Analysis:** `auc_report.csv`, `roc_curves.png` (One-vs-Rest macro & weighted AUC)
* **Feature Space Visualization:** `tsne_plot.png` & `tsne_embeddings.csv` (2D t-SNE projection of the penultimate `embedding` layer)
* **Error Analysis:** `misclassified_examples.csv` listing false predictions and confidence scores
* **Complexity & Efficiency:** `model_performance_complexity.csv` (trainable/fixed parameters, disk size in MB, training time, and single-sample inference latency in ms)
* **Pareto Analysis (`outputs/deep_model_comparison/`):** Identifies non-dominated models balancing accuracy, parameter count, model size, and inference latency for edge vs. high-accuracy deployment scenarios.