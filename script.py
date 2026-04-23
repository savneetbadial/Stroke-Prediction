# Pre-processing script
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# 1. Load the dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# 2. Data cleaning: handle missing values
# The bmi column has missing values use the median to fill them
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# 3. Categorical encoding
# Drop 'id'
df = df.drop(columns=['id'])

# Label encoding for binary features
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['ever_married'] = le.fit_transform(df['ever_married'])
df['Residence_type'] = le.fit_transform(df['Residence_type'])

# Encoding for nominal features:work_type, smoking_status
df = pd.get_dummies(df, columns=['work_type', 'smoking_status'])

# 4. Feature and target split
X = df.drop(columns=['stroke'])
y = df['stroke']

# 5. Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. PCA Implementation
# Reducing features while retaining 95% of the variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 7. Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

print(f"Original feature count: {X.shape[1]}")
print(f"Reduced feature count after PCA: {X_pca.shape[1]}")