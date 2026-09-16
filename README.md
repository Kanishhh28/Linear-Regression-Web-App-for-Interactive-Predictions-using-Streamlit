# Medical Insurance Charges Prediction Web App

A Streamlit-based machine learning application that predicts medical insurance charges from demographic, lifestyle, and geographic information. The project uses Linear Regression with preprocessing for categorical variables and provides an interactive web interface for making individual predictions.

## Overview

This project implements an end-to-end machine learning workflow:

1. Load the medical insurance dataset.
2. Separate input features from the target variable (`charges`).
3. Encode categorical variables.
4. Split the dataset into training and testing sets.
5. Train a Linear Regression model.
6. Evaluate the model using R-squared.
7. Save the trained preprocessing-and-regression pipeline as `medical.pkl`.
8. Load the saved model in a Streamlit application.
9. Accept user inputs through an interactive form.
10. Predict the expected medical insurance charge.

The project contains both an exploratory notebook implementation and a cleaner training script used to create the model consumed by the Streamlit application.

---

## Features

### Machine Learning

- Linear Regression for continuous-value prediction
- Train/test split with an 80/20 ratio
- Categorical feature encoding
- Pipeline-based preprocessing and model training
- R-squared evaluation in the original notebook
- Serialized trained model using Python pickle

### Interactive Web Application

The Streamlit interface accepts:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

After the user clicks **Predict**, the application sends the input through the saved machine learning pipeline and displays the predicted insurance charge.

---

## Dataset

The project uses the `medical_insurance.csv` dataset.

### Dataset size

The dataset contains:

- 1,338 records
- 7 columns

### Columns

| Column | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the individual |
| `sex` | Categorical | Sex of the individual |
| `bmi` | Numeric | Body Mass Index |
| `children` | Numeric | Number of children/dependents |
| `smoker` | Categorical | Smoking status |
| `region` | Categorical | Geographic region |
| `charges` | Numeric | Medical insurance charge; prediction target |

The target variable is:

```text
charges
```

The input features are:

```text
age
sex
bmi
children
smoker
region
```

---

## Machine Learning Pipeline

The current production-oriented training script is `medical_insurance.py`.

### 1. Load the dataset

```python
df = pd.read_csv("medical_insurance.csv")
```

### 2. Separate features and target

```python
X = df.drop("charges", axis=1)
y = df["charges"]
```

### 3. Identify categorical features

The following columns are categorical:

```python
categorical_features = ['sex', 'smoker', 'region']
```

### 4. One-hot encode categorical variables

The project uses `ColumnTransformer` with:

```python
OneHotEncoder(drop='first')
```

This converts categorical values into numerical features while dropping the first category from each categorical variable.

The numerical variables are retained unchanged using:

```python
remainder='passthrough'
```

### 5. Build the pipeline

The preprocessing and regression model are combined into one scikit-learn `Pipeline`:

```text
Input Data
    |
    v
ColumnTransformer
    |
    +-- OneHotEncoder for categorical columns
    |
    +-- Pass numerical columns unchanged
    |
    v
LinearRegression
    |
    v
Predicted Charges
```

This is important because the same preprocessing used during training is automatically applied to user input during prediction.

### 6. Train/test split

The training script uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Therefore:

- 80% of the data is used for training.
- 20% is used for testing.
- `random_state=42` makes the split reproducible.

### 7. Train Linear Regression

```python
pipeline.fit(X_train, y_train)
```

### 8. Save the trained pipeline

The trained pipeline is serialized to:

```text
medical.pkl
```

using Python's `pickle` module.

---

## Streamlit Application

The Streamlit application is implemented in:

```text
medical_.py
```

The application loads the serialized model:

```python
with open('medical.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
```

It then creates interactive input widgets for each model feature.

### Input interface

```text
Age
Sex
BMI
Number of Children
Smoker
Region
        |
        v
     Predict
        |
        v
Medical Insurance Charge
```

The submitted values are placed into a pandas DataFrame using the same feature names expected by the trained pipeline.

The prediction is then generated with:

```python
prediction = model.predict(input_data)[0]
```

The result is displayed as a dollar amount rounded to two decimal places.

---

## Project Structure

The original ZIP contains the following files:

```text
Linear-Regression-Web-App-for-Interactive-Predictions-using-Streamlit-main/
│
├── .DS_Store
├── medical.pkl
├── medical_.py
├── medical_insu_fds.ipynb
├── medical_insurance.csv
└── medical_insurance.py
```

### Recommended final structure

```text
medical-insurance-prediction/
│
├── README.md
├── medical_.py
├── medical_insurance.py
├── medical_insurance.csv
├── medical.pkl
├── requirements.txt
└── .gitignore
```

---

## File Descriptions

### `medical_.py`

The Streamlit frontend/application.

Responsibilities:

- Load the trained model.
- Display the application interface.
- Collect user input.
- Construct a DataFrame from the input.
- Run the prediction.
- Display the predicted insurance charge.
- Display an error message if prediction fails.

### `medical_insurance.py`

The main model-training script.

Responsibilities:

- Load the dataset.
- Separate features and target.
- Define categorical columns.
- Build the preprocessing transformer.
- Build the Linear Regression pipeline.
- Split the data.
- Train the pipeline.
- Save the trained pipeline as `medical.pkl`.

This is the preferred reproducible training script in the project.

### `medical_insu_fds.ipynb`

An earlier/exploratory notebook containing the original machine learning workflow.

The notebook performs:

- Dataset loading
- Feature/target separation
- Label encoding of `sex` and `smoker`
- One-hot encoding of `region`
- Train/test splitting
- Linear Regression training
- Test prediction
- R-squared calculation
- Pickle serialization

The notebook is useful as a record of the development/learning process but is not required to run the Streamlit application.

### `medical.pkl`

Serialized trained scikit-learn pipeline.

It contains the preprocessing and Linear Regression model used by the Streamlit application.

### `medical_insurance.csv`

Training dataset containing demographic, lifestyle, regional, and insurance-charge information.

### `.DS_Store`

macOS Finder metadata. It is not part of the application.

---

## Redundant Files and Cleanup

After reviewing the complete project, the following cleanup is recommended.

### Remove: `.DS_Store`

This is macOS-generated metadata and should not be committed to GitHub.

Add this to `.gitignore`:

```text
.DS_Store
```

### Keep but optionally archive: `medical_insu_fds.ipynb`

The notebook is not required for running the application because the trained model is already stored in `medical.pkl` and the reproducible training process is available in `medical_insurance.py`.

If the repository is intended to showcase the machine learning learning process, keep the notebook.

If the repository is intended to be a clean application/deployment repository, it can be removed or moved to a `notebooks/` directory.

### Do not remove: `medical_insurance.py`

This is the clean model-training script and should be retained. It provides a reproducible way to rebuild `medical.pkl`.

### Do not remove: `medical_.py`

This is the Streamlit application and is required to run the interactive prediction interface.

### Do not remove: `medical_insurance.csv`

The dataset is required to retrain the model using the provided training script.

### Do not remove: `medical.pkl`

The Streamlit application requires the serialized model unless the application is changed to train/load the model in another way.

---

## Recommended Renaming

The current filename:

```text
medical_.py
```

is functional but not descriptive.

A clearer name would be:

```text
app.py
```

or:

```text
streamlit_app.py
```

For example:

```text
medical-insurance-prediction/
├── app.py
├── medical_insurance.py
├── medical_insurance.csv
├── medical.pkl
├── requirements.txt
└── README.md
```

If you rename the file, the command used to start Streamlit should use the new filename.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

Create a `requirements.txt` containing:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
```

Then install:

```bash
pip install -r requirements.txt
```

The Streamlit application itself imports `streamlit`, `pandas`, and `pickle`. The training workflow additionally uses pandas and scikit-learn; the notebook also imports NumPy and Matplotlib.

---

## Running the Application

From the project directory:

```bash
streamlit run medical_.py
```

If the application file is renamed to `app.py`:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open that address in a browser.

---

## Running the Training Pipeline

To rebuild the model from the dataset:

```bash
python medical_insurance.py
```

The script will:

1. Read `medical_insurance.csv`.
2. Split the data into training and testing sets.
3. Build the categorical preprocessing transformer.
4. Train the Linear Regression model.
5. Save the complete pipeline to `medical.pkl`.

Expected terminal message:

```text
Model trained and saved as medical.pkl
```

The exact printed message in the current script includes a check-mark character.

---

## Prediction Workflow

When a user enters information in the web application:

```text
User Input
   |
   +-- Age
   +-- Sex
   +-- BMI
   +-- Children
   +-- Smoker
   +-- Region
   |
   v
Pandas DataFrame
   |
   v
Saved Scikit-Learn Pipeline
   |
   +-- One-Hot Encoding
   |
   +-- Numerical Features Passed Through
   |
   v
Linear Regression
   |
   v
Predicted Insurance Charges
```

The preprocessing pipeline is embedded inside `medical.pkl`, so the Streamlit application does not need to manually encode the categorical input values.

---

## Model Evaluation

The original notebook evaluates the Linear Regression model using the R-squared metric.

```python
from sklearn.metrics import r2_score

r2 = r2_score(Y_test, y_pred)
print("R-squared:", r2)
```

R-squared measures the proportion of variation in the target variable explained by the model on the test set.

The current Streamlit application does not display the evaluation metric to users.

---

## Important Implementation Difference

There are two versions of the training workflow in the repository.

### Original notebook

The notebook manually performs:

```text
Label Encoding
    |
One-Hot Encoding
    |
Train/Test Split
    |
Linear Regression
```

It applies `LabelEncoder` to `sex` and `smoker` and `OneHotEncoder` to `region`.

### Current training script

`medical_insurance.py` uses a more consistent pipeline:

```text
Categorical Features
        |
OneHotEncoder(drop='first')
        |
ColumnTransformer
        |
LinearRegression
```

The entire preprocessing/model pipeline is saved together.

For the final project, the **`medical_insurance.py` pipeline should be treated as the primary training implementation**, because it keeps preprocessing and prediction together and reduces the chance of applying inconsistent transformations between training and inference.

---

## Model Artifact Compatibility

`medical.pkl` is a serialized scikit-learn object. It should ideally be loaded using a compatible Python and scikit-learn environment.

The supplied model was serialized with an older scikit-learn version than the environment used during inspection. This produced scikit-learn version compatibility warnings when the pickle was inspected.

For reproducibility, record the exact Python and package versions used to train the final model.

A recommended `requirements.txt` for a reproducible version should pin the versions actually used to train `medical.pkl`.

If the original training environment is unavailable, retrain the model with the current environment and regenerate `medical.pkl`.

---

## Limitations

- The model is a Linear Regression model and may not capture complex nonlinear relationships in insurance charges.
- The project uses a relatively small tabular dataset.
- No feature scaling is applied to the numerical variables.
- Model evaluation is limited to R-squared in the notebook.
- There is no automated test suite.
- The application does not provide prediction intervals or uncertainty estimates.
- The application is intended for demonstration/educational use rather than actual insurance pricing.
- The model should not be interpreted as a clinical or financial decision-making system.
- The model artifact is dependent on compatible Python/scikit-learn versions.

---

## Possible Improvements

### Machine Learning

- Compare Linear Regression with:
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Add MAE and RMSE alongside R-squared.
- Perform cross-validation.
- Tune model hyperparameters where applicable.
- Analyze residuals.
- Investigate feature importance or model coefficients.
- Consider transformations for highly skewed insurance charges.

### Application

- Add input validation and more descriptive error messages.
- Display model evaluation metrics.
- Add an explanation of the factors used by the model.
- Add prediction ranges or uncertainty estimates.
- Improve UI layout and formatting.
- Add charts showing how predicted charges vary with selected inputs.

### Project Engineering

- Add `requirements.txt`.
- Add `.gitignore`.
- Remove `.DS_Store`.
- Use relative paths consistently.
- Rename `medical_.py` to `app.py` or `streamlit_app.py`.
- Pin package versions.
- Add automated tests.
- Add a clear deployment configuration.

---

## Suggested `.gitignore`

```text
__pycache__/
*.py[cod]
*.pyo
.venv/
venv/
env/
.env
.DS_Store
.ipynb_checkpoints/
```

---

## Security and Data Considerations

The application does not currently require API keys or external credentials.

However, if the application is later extended with external services, credentials should be stored in environment variables rather than committed to the repository.

The dataset contains individual-level demographic and insurance information. If this project is redistributed or deployed publicly, verify that the dataset is appropriate for public use and does not contain sensitive personal information.

---

## Use Cases

This project demonstrates how machine learning can be integrated into an interactive web application for:

- Predictive analytics
- Regression modeling
- Tabular data preprocessing
- Categorical feature encoding
- Model serialization
- Interactive ML demonstrations
- Rapid prototyping with Streamlit

It is particularly suitable as an educational project demonstrating the transition from a machine learning notebook to a usable prediction application.

---

## End-to-End Architecture

```text
                    medical_insurance.csv
                              |
                              v
                    medical_insurance.py
                              |
                    +---------+---------+
                    |                   |
                    v                   v
              Preprocessing      Train/Test Split
                    |                   |
                    +---------+---------+
                              |
                              v
                       Linear Regression
                              |
                              v
                         medical.pkl
                              |
                              v
                         medical_.py
                              |
                              v
                     Streamlit Web App
                              |
                              v
                         User Inputs
                              |
                              v
                   Insurance Charge Prediction
```

---

## Final Recommended Repository

For a clean GitHub repository, the recommended final structure is:

```text
medical-insurance-prediction/
│
├── README.md
├── app.py
├── medical_insurance.py
├── medical_insurance.csv
├── medical.pkl
├── requirements.txt
└── .gitignore
```

Optional:

```text
└── notebooks/
    └── medical_insu_fds.ipynb
```

The notebook can be retained under `notebooks/` if you want the repository to document the original experimentation and learning process.

---

## Author

Kanishhh28

GitHub:
https://github.com/Kanishhh28
