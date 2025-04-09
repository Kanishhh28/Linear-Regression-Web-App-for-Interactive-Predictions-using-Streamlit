import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
df = pd.read_csv("medical_insurance.csv")

# Split features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Define categorical features to encode
categorical_features = ['sex', 'smoker', 'region']

# Column transformer with one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'  # Keep numerical columns as-is
)

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save pipeline
with open("medical.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as medical.pkl")
