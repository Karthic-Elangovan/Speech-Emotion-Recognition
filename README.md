# Speech Emotion Recognition (SER) using LSTM

This project implements a Speech Emotion Recognition (SER) system using Long Short-Term Memory (LSTM) networks. The system processes and classifies speech signals to detect embedded emotions from audio files.


## Project Overview

Speech Emotion Recognition (SER) is a collection of methodologies that process and classify speech signals to detect embedded emotions. This project focuses on implementing an SER system using LSTM networks, which are particularly effective for sequential data like audio signals due to their ability to retain context information over longer time spans.

Key features:
- Processes four popular English emotion datasets (Crema, Ravdess, Savee, and Tess)
- Uses Mel-frequency cepstral coefficients (MFCC) for feature extraction
- Implements data augmentation techniques (noise injection and time stretching)
- Visualizes audio waveforms and amplitude envelopes
- Trains an LSTM model for emotion classification

## Datasets

The project uses four English emotion datasets:

1. **Crema Dataset**
   - Emotions: sad, angry, disgust, fear, happy, neutral
   - Filename format indicates emotion (e.g., `1028_TSI_DIS_XX.wav` for disgust)

2. **Ravdess Dataset**
   - Emotions: neutral, calm, happy, sad, angry, fear, disgust, surprise
   - Complex filename encoding (e.g., `03-01-08-01-01-01-02.wav`)

3. **Savee Dataset**
   - Emotions: anger, disgust, fear, happiness, neutral, sadness, surprise
   - Prefix letters indicate emotion (e.g., 'a' = anger)

4. **Tess Dataset**
   - Emotions: fear, angry, disgust, neutral, sad, surprise, happy
   - Emotion label in filename (e.g., `YAF_fear.wav`)

The combined dataset contains over 7,000 audio samples across these emotion categories.

## Methodology

### Feature Extraction
- **MFCC (Mel-frequency cepstral coefficients):** Extracted from audio files using librosa
- **Amplitude Envelope:** Captures signal intensity variations over time
- **Short-Time Fourier Transform (STFT):** Analyzes specific segments of the speech signal

### Data Augmentation
- **Noise Injection:** Adds random noise to make the model more robust
- **Time Stretching:** Alters audio duration without changing pitch

### Model Architecture
The LSTM model consists of:
1. LSTM layer (256 units)
2. Dropout layer (0.2)
3. Dense layer (128 units, ReLU activation)
4. Dropout layer (0.2)
5. Dense layer (64 units, ReLU activation)
6. Dropout layer (0.2)
7. Output layer (6 units, softmax activation)

### Training
- Optimizer: Adam
- Loss: Categorical crossentropy
- Batch size: 64
- Epochs: 500
- Validation split: 20%

## Implementation

### Data Preprocessing
1. Load audio files from all datasets
2. Extract emotion labels from filenames
3. Combine datasets and standardize emotion labels
4. Apply data augmentation (noise and stretch)

### Feature Engineering
1. Extract MFCC features (40 coefficients)
2. Calculate amplitude envelopes
3. Visualize waveforms and features

### Model Training
1. Build LSTM model architecture
2. Compile with Adam optimizer
3. Train with early stopping and learning rate scheduling
4. Evaluate on validation set

## Usage

To use this project:

1. Clone the repository
2. Install dependencies (see below)
3. Place datasets in the appropriate directory structure
4. Run the Jupyter notebook




## Dependencies

- Python 3.x
- TensorFlow/Keras
- Librosa
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- IPython

Install dependencies with:
```bash
pip install tensorflow librosa numpy pandas matplotlib seaborn scikit-learn ipython
```
