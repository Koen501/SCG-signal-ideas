# SCG-signal-ideas
Blood Pressure Prediction Using SCG Signals Based on Machine Learning Methods 
# Non-invasive Continuous Blood Pressure Prediction Based on SCG Signals (Machine-learning for BP from SCG)

Project Overview
---
This repository implements and reproduces a blood pressure prediction method based on thoracic wall vibration signals (Seismocardiogram, SCG). The objective is to estimate systolic blood pressure (SBP) and diastolic blood pressure (DBP) using machine learning models (linear regression and Support Vector Regression, SVR) applied to SCG signals collected by wearable devices. This provides a non-invasive, continuously monitorable blood pressure measurement solution, addressing the limitations of the cuff method for long-term monitoring and the susceptibility of PTT to interference.

Paper Abstract (Summary)
---
Hypertension is a prevalent chronic disease in China, and accurate blood pressure measurement is crucial for health management. This work proposes a blood pressure prediction method based on SCG signals: SCG is collected from subjects at rest, baseline drift is removed using Butterworth low-pass filtering, and AO peaks (aortic valve opening events) are localized. After extracting electro-mechanical features, both linear regression and support vector regression (SVR) are employed for modeling. Results (measured by MAE):
- Linear regression: DBP MAE = 4.29 mmHg; SBP MAE = 13.4 mmHg
- SVR: DBP MAE = 1.8 mmHg; SBP MAE = 3.2 mmHg

According to IEEE 1708-2014 standards (DBP MAE ≤ 2.6 mmHg, SBP MAE ≤ 3.2 mmHg), SVR achieved superior results. With further dataset expansion, this method holds promise for non-invasive continuous blood pressure monitoring.

Keywords
---
machine learning, multimodal fusion, wearable devices, SCG signal, blood pressure prediction

Repository Highlights
---
- Utilizes SCG (Strain-Coupled Heartbeat Signal) as input, offering a mechanical perspective distinct from traditional ECG/PTT
- Streamlined preprocessing pipeline: Butterworth low-pass filtering + AO peak detection
- Performance comparison between linear regression and SVR shows SVR excels on this dataset while meeting IEEE 1708-2014 thresholds
- Lightweight model with high computational efficiency, suitable for deployment and inference on mobile/wearable platforms

Dependencies (Recommended)
---
Recommended Python environment with common packages (example):
- Python 3.8+
- numpy, scipy
- pandas
- scikit-learn
- matplotlib / seaborn (for visualization)
- librosa / peakutils / biosppy (optional, for peak detection and signal processing)

Can use requirements.txt (example):
```
numpy
scipy
pandas
scikit-learn
matplotlib
seaborn
```

Data Description and Preprocessing
---
(Describe data format and acquisition locations; modify based on actual data storage paths)
- Data location: `data/` (recommended to organize subfolders by subject ID or record ID)
- Each record should contain: SCG signal (time series), corresponding SBP/DBP labels (mmHg), sampling rate information
- Preprocessing steps:
  1. Remove baseline drift using Butterworth low-pass filtering (filter parameters configurable)
  2. Standardization/Noise reduction (Optional: Bandpass filtering, artifact removal)
  3. AO peak detection and extraction of cardiac cycles (or windowed segments)
  4. Feature extraction (Time-domain, frequency-domain, wavelet features, or direct end-to-end learning using segments)

Feature Examples
---
- Peak-to-peak interval (AO to AO)
- Peak amplitude, rise/fall slope
- Energy within period, bandwidth energy distribution
- Statistics: mean, standard deviation, skewness, kurtosis
(Extensible: wavelet packet or frequency band energy, etc.)

Models and Evaluation
---
- Comparison Models:
  - Linear Regression (baseline)
  - Support Vector Regression (SVR, RBF or linear kernel)
- Training/Validation Recommendations:
  - Cross-validation (k-fold) or subject-wise split for generalization assessment
  - Metrics: MAE (Mean Absolute Error) primary, supplemented by RMSE, R²
- Reference Standard: IEEE 1708-2014 (DBP MAE ≤ 2.6 mmHg, SBP MAE ≤ 3.2 mmHg)

Key Results (Examples)
---
- Linear Regression: DBP MAE = 4.29 mmHg; SBP MAE = 13.4 mmHg
- SVR: DBP MAE = 1.8 mmHg; SBP MAE = 3.2 mmHg (meets IEEE 1708-2014 upper limit for SBP and exceeds DBP requirement)

How to Reproduce Experiments (Example Steps)
---
1. Place data in `data/` folder, ensuring each record contains SCG and corresponding label
2. Set up environment and install dependencies:
   - python -m venv venv && source venv/bin/activate
   - pip install -r requirements.txt
3. Run preprocessing script (example):
   - python preprocess.py --input data/ --output processed/
4. Train the model (example):
   - python train.py --data processed/ --model svr --cv 5 --out models/
5. Evaluate and generate results:
   - python evaluate.py --model models/svr.pkl --data processed/ --metrics mae rmse

Note: The script names and parameters above are examples; please adjust commands based on your repository's actual files or add corresponding scripts to this repository.

Recommended Repository Structure
---
- data/                    # Raw data (do not upload sensitive/protected data)
- processed/               # Preprocessed data
- notebooks/               # Experiment and visualization Notebooks
- src/                     # Code: Preprocessing, Features, Training, Evaluation
  - preprocess.py
  - feature_extraction.py
  - train.py
  - evaluate.py
- models/                  # Stored trained models
- requirements.txt
- README.md

Privacy and Compliance
---
- If data contains personally identifiable information (PII) or medically sensitive information, strictly adhere to ethical and privacy protection standards. Remove or anonymize personal information before sharing.
- Private data should not be directly pushed to public repositories; use synthetic/anonymized data or provide data loading interfaces for users to prepare their own data when sharing results or code

Future Directions
---
- Expand the dataset to include subjects of varying ages, genders, medical histories, and activity levels to enhance model robustness
- Employ more complex models (deep learning, time series models) for end-to-end learning
- Multimodal fusion: Jointly model SCG with ECG/PPG signals to improve accuracy
- Optimize and deploy real-time inference models on wearable devices

Citations and Acknowledgments
---
If citing this project in academic work, please reference the corresponding paper or add bibtex entries (if available) to the README. We extend our gratitude to all volunteers and researchers involved in data collection and evaluation.

Contribution Guidelines
---
- Submit issues and pull requests
- To add data or scripts, first describe proposed changes in an issue, specifying data sources and anonymization methods

License
---

Contact
---
yguo32022@gmail.com

Translated with DeepL.com (free version)
