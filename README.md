# AI-Based Electrical Insulation Degradation Classification Using Leakage Current Patterns

## Overview
A reproducible educational capstone that extracts time- and frequency-domain features from leakage-current windows and trains two classifiers:
- Random Forest
- Histogram-based Gradient Boosting

The supplied dataset is **synthetically simulated** to make the project runnable without specialized high-voltage equipment. Its three labels are Healthy, Moderately Degraded, and Severely Degraded. Do not present these records as measured laboratory data.

## Project structure
```
data/synthetic_leakage_current.csv   # generated synthetic signal windows
src/features.py                     # signal feature extraction
src/train_model.py                  # training + unseen test evaluation
src/dashboard.py                    # Streamlit dashboard
models/                             # created by training script
reports/evaluation_metrics.json      # created by training script
```

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
streamlit run src/dashboard.py
```

In Google Colab, upload/unzip the project, run `pip install -r requirements.txt`, then `!python src/train_model.py`. For the dashboard, local execution is easiest; Colab needs a separate tunneling setup.

## Signal pipeline
1. Parse each signal window and reject non-finite/invalid samples.
2. Extract mean, RMS, peak, standard deviation, mean absolute current, crest factor, skewness, kurtosis, zero-crossing rate, dominant frequency, spectral centroid and high-frequency power ratio.
3. Split windows into training and test sets using a stratified 75/25 split.
4. Fit both classifiers only on training rows.
5. Report accuracy, class-wise precision/recall/F1 and confusion matrices on unseen test rows.
6. Use the dashboard to view a waveform, FFT magnitude spectrum, features, predicted class and model probabilities.

## Important limitations
- Simulated signals do not establish real-world diagnostic performance.
- The labels are generated from the simulation recipe, so high scores may reflect the recipe rather than actual insulation physics.
- Real deployment needs a documented measured dataset, expert-verified labels, device-independent testing, calibration, uncertainty handling, and electrical safety review.
- Current units are arbitrary simulated units, not amperes.
- This tool is an educational prototype, not a safety-rated instrument.
