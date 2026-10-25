# Structural Crack Detection using Denoising Autoencoders & ResNet50

A deep learning pipeline and Flask web application for automated detection and classification of structural cracks in concrete infrastructure (surfaces, bridges, walls, and pavements).

## Architecture & Methodology
1. **Denoising Autoencoder (DAE):**
   - Convolutional autoencoder trained to filter visual noise, surface irregularities, and environmental artifacts from surface imagery.
   - Reconstructs clean 64x64 feature maps while preserving subtle fracture boundaries.
2. **Transfer Learning Classifier (ResNet50):**
   - Deep residual network pre-trained on ImageNet and fine-tuned for binary classification (`Crack` vs. `No Crack`).
   - Evaluated both as a raw baseline and in combination with denoising preprocessing.
3. **Interactive Flask Web Application:**
   - Real-time image upload, inference, and visualization of crack probabilities.

## Performance Benchmark
Evaluated on holdout test splits:
- **ResNet50 Baseline:** 93.46% Test Accuracy (Loss: 0.1958)
- **ResNet50 + Denoised:** 87.34% Test Accuracy (Loss: 0.3438)

## Project Structure
```
Structural_Crack_Detection/
|-- app.py                     # Flask web application & inference server
|-- requirements.txt           # Project Python dependencies
|-- src/
|   |-- prepare_dataset.py     # Train/test split preparation
|   |-- train_autoencoder.py   # Denoising autoencoder training
|   |-- train_resnet.py        # ResNet50 baseline training
|   |-- train_resnet_with_denoising.py # Denoised pipeline training
|   |-- evaluate.py            # Comparative evaluation & metrics generation
|   |-- predict.py             # Single-image inference CLI
|-- models/
|   |-- autoencoder.keras      # Trained convolutional autoencoder weights (3.88 MB)
|-- outputs/
|   |-- metrics/               # Evaluation results
|   |-- plots/                 # Training curves and comparison graphs
|   |-- sample_predictions/    # Visual prediction samples
|-- templates/
|   |-- index.html             # Flask web application UI
```

## Setup & Installation

### 1. Clone & Environment Setup
```bash
git clone https://github.com/chanu08/structural-crack-detection.git
cd structural-crack-detection
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Web Dashboard
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

### 3. Model Training & Evaluation
```bash
# Train Denoising Autoencoder
python src/train_autoencoder.py

# Train ResNet Classifier
python src/train_resnet.py

# Evaluate Models
python src/evaluate.py
```

## Dataset
This project is built using the **SDNET2018** structural crack dataset. Due to storage constraints (>1.5 GB), raw and partitioned image datasets are excluded from this repository and should be downloaded separately into `dataset/`.
