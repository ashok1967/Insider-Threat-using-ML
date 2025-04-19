# Insider Threat Detection - CMU Dataset
# Complete pipeline from data loading to deep learning evaluation using Dask for scalable big data processing

import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input, RepeatVector, Dropout, TimeDistributed, Reshape
from keras.callbacks import History
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# --- Data Paths ---
DATA_PATH = "./data/"

# --- Load large files using Dask ---
print("Loading large CSVs with Dask...")
email = dd.read_csv(DATA_PATH + "email.csv", usecols=['user','date','activity','size'], assume_missing=True)
file_activity = dd.read_csv(DATA_PATH + "file.csv", usecols=['user','date','activity'], assume_missing=True)
logon = dd.read_csv(DATA_PATH + "logon.csv", assume_missing=True)
http = dd.read_csv(DATA_PATH + "http.csv", usecols=['user','date','activity'], assume_missing=True)

# --- Load smaller files with Pandas ---
print("Loading small CSVs with Pandas...")
decoy_file = pd.read_csv(DATA_PATH + "decoy_file.csv")
device = pd.read_csv(DATA_PATH + "device.csv")
ldap = pd.read_csv(DATA_PATH + "LDAP.csv")
psych = pd.read_csv(DATA_PATH + "pschrometic.csv")

# --- Preprocessing Dask DataFrames ---
print("Preprocessing Dask-based datasets...")
email['date'] = dd.to_datetime(email['date'], errors='coerce')
file_activity['date'] = dd.to_datetime(file_activity['date'], errors='coerce')
logon['date'] = dd.to_datetime(logon['date'], errors='coerce')
http['date'] = dd.to_datetime(http['date'], errors='coerce')

email = email.dropna(subset=['date'])
file_activity = file_activity.dropna(subset=['date'])
logon = logon.dropna(subset=['date'])
http = http.dropna(subset=['date'])

# --- Aggregation ---
print("Aggregating activities...")
email_agg = email.groupby('user').agg({
    'activity': 'count',
    'size': 'mean'
}).rename(columns={'activity': 'email_count', 'size': 'avg_email_size'}).compute()

file_agg = file_activity.groupby('user').agg({'activity': 'count'}).rename(columns={'activity': 'file_activity_count'}).compute()
logon_agg = logon.groupby('user').agg({'activity': 'count'}).rename(columns={'activity': 'logon_count'}).compute()
http_agg = http.groupby('user').agg({'activity': 'count'}).rename(columns={'activity': 'http_activity_count'}).compute()

# --- Encode Categorical ---
print("Encoding categorical features...")
ldap['role'] = LabelEncoder().fit_transform(ldap['role'])
ldap['business_unit'] = LabelEncoder().fit_transform(ldap['business_unit'].astype(str))

# --- Merge All Data ---
print("Merging all features into unified DataFrame...")
df_merged = ldap.merge(email_agg, left_on='user_id', right_index=True, how='left')
df_merged = df_merged.merge(file_agg, left_on='user_id', right_index=True, how='left')
df_merged = df_merged.merge(logon_agg, left_on='user_id', right_index=True, how='left')
df_merged = df_merged.merge(http_agg, left_on='user_id', right_index=True, how='left')
df_merged = df_merged.merge(psych, on='user_id', how='left')
df_merged.fillna(0, inplace=True)

# --- Risk Label Placeholder ---
df_merged['label'] = (df_merged['file_activity_count'] > 50).astype(int)

# --- Feature Engineering ---
print("Performing additional feature engineering...")
df_merged['behavior_score'] = (
    df_merged['email_count'] * 0.2 +
    df_merged['file_activity_count'] * 0.4 +
    df_merged['logon_count'] * 0.2 +
    df_merged['http_activity_count'] * 0.2
)

# --- Normalize Features ---
print("Normalizing features...")
X = df_merged.drop(columns=['user_id', 'employee name', 'email', 'label'])
y = df_merged['label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Handle Imbalanced Classes ---
print("Applying SMOTE for class balancing...")
X_bal, y_bal = SMOTE().fit_resample(X_scaled, y)

# --- Train/Test Split ---
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.3, random_state=42)

print("Feature engineering, normalization, and data split complete.")

# --- Feature Importance Plot using Random Forest ---
print("Generating feature importance plot...")
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
feature_names = df_merged.drop(columns=['user_id', 'employee name', 'email', 'label']).columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()
