import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

# Function to calculate Quadratic Kappa
def quadratic_kappa(y_true, y_pred):
    """Compute Cohen Kappa Score with weights='quadratic'."""
    return cohen_kappa_score(y_true, np.round(y_pred), weights='quadratic')

# Load data
data = pd.read_csv('/home/quoc/works/FinanceLearn/Data/Train_data_merge.csv')

# Data preprocessing
object_columns = data.select_dtypes(include=['object']).columns
for col in object_columns:
    data[col] = data[col].astype('category')

imputer = KNNImputer(n_neighbors=71)
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputed_data = imputer.fit_transform(data[numeric_cols])
train_imputed = pd.DataFrame(imputed_data, columns=numeric_cols)
train_imputed['sii'] = train_imputed['sii'].round().astype(int)
for col in data.columns:
    if col not in numeric_cols:
        train_imputed[col] = data[col]
data = train_imputed

def handle_category_data(data):
    df_encoded = data.copy()
    for column in data.columns:
        if df_encoded[column].dtype == 'object' or df_encoded[column].dtype.name == 'category':
            df_encoded = pd.get_dummies(df_encoded, columns=[column], drop_first=True)
    return df_encoded

data = handle_category_data(data)

X = data.drop(columns=['sii'])  # Replace 'sii' with your target column
y = data['sii']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Objective function for DecisionTreeRegressor
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'random_state': 42
    }

    model = DecisionTreeRegressor(**params)
    
    # Use Cross-Validation to compute Quadratic Kappa Score
    scorer = make_scorer(quadratic_kappa, greater_is_better=True)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring=scorer).mean()
    
    return score

# Optimize with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)  # Try with 50 trials

# Results
print("Best parameters found for DecisionTree:", study.best_params)
print("Best quadratic kappa score for DecisionTree:", study.best_value)

# Evaluate the best model on the test set
best_params = study.best_params
best_model = DecisionTreeRegressor(**best_params)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
test_kappa_score = quadratic_kappa(y_test, y_pred)
print("Quadratic Kappa Score on test set (DecisionTree):", test_kappa_score)
