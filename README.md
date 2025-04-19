# Insider-Threat-using-ML
This is the code developed for Insider Threat detection using Machine Learning and Deeep Learning
# Insider Threat Detection

A complete machine learning and deep learning pipeline using the CMU Insider Threat Dataset.  
Utilizes scalable data handling via Dask and applies ensemble learning and deep neural networks for anomaly and threat detection.

## Features
- Efficient handling of multi-GB files using Dask
- Feature engineering across multiple user activities
- Machine Learning: Random Forest, SVM, MLP, XGBoost
- Deep Learning: DNN, LSTM Autoencoder, BiLSTM, CNN+LSTM
- Handles class imbalance with SMOTE
- Evaluates using Accuracy, Precision, Recall, F1-Score
- Feature importance visualization
- Epoch vs. Accuracy plots

## Structure
- `Insider_Threat_Detection.ipynb`: Main notebook
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Setup

```bash
pip install -r requirements.txt
jupyter notebook Insider_Threat_Detection.ipynb
```

## Dataset
Ensure the CMU Insider Threat dataset r6.2 files are placed in a `./data/` folder at the root level.

