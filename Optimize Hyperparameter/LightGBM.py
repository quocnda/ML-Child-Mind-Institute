import optuna
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
# Tắt log của Optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

# Hàm tính Quadratic Kappa
def quadratic_kappa(y_true, y_pred):
    """Hàm tính Cohen Kappa Score với weights='quadratic'."""
    return cohen_kappa_score(y_true, np.round(y_pred), weights='quadratic')

# Đọc dữ liệu
data = pd.read_csv('/home/quoc/works/FinanceLearn/Data/Train_data_merge.csv')
object_columns = data.select_dtypes(include=['object']).columns
for col in object_columns:
    data[col] = data[col].astype('category')

# Impute missing values
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
    # Define the fixed set of seasons
    all_categories = ['Spring', 'Summer', 'Fall', 'Winter']

    df_encoded = data.copy()
    encoder = OneHotEncoder(categories=[all_categories],  # Đảm bảo 4 chiều
                             sparse_output=False,         # Đầu ra dạng mảng dày
                             handle_unknown='ignore',     # Bỏ qua giá trị ngoài danh mục
                             dtype=int)                   # Đầu ra kiểu int

    for column in data.columns:
        if df_encoded[column].dtype == 'object' or df_encoded[column].dtype.name == 'category':
            # Fit and transform the column
            encoded_array = encoder.fit_transform(df_encoded[[column]])
            
            # Generate column names
            encoded_columns = [f"{column}_{season}" for season in all_categories]
            
            # Add the encoded data to the DataFrame
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=data.index)
            df_encoded = pd.concat([df_encoded.drop(column, axis=1), encoded_df], axis=1)
    
    return df_encoded
data = handle_category_data(data)

# Tách dữ liệu
X = data.drop(columns=['sii'])
y = data['sii']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hàm mục tiêu cho Optuna
def objective(trial):
    # Hyperparameter search space
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 50.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 50.0, log=True),
        'verbosity': -1
}


    # Train LightGBM
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    # Predict trên tập test
    y_pred = model.predict(X_test)
    
    # Tính Quadratic Kappa Score
    score = quadratic_kappa(y_test, y_pred)
    
    return score

# Tối ưu hóa với Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=150)  # Tối ưu hóa với 150 lần thử nghiệm

# Kết quả tốt nhất
print("Best params for LightGBM:", study.best_params)
print("Best quadratic kappa score on validation set:", study.best_value)

# Đánh giá mô hình tốt nhất trên tập kiểm tra
best_params = study.best_params
best_model = LGBMRegressor(**best_params, verbosity=-1)  # Tắt log của LightGBM
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
test_kappa_score = quadratic_kappa(y_test, y_pred)
print("Quadratic Kappa Score on test set (LightGBM):", test_kappa_score)
