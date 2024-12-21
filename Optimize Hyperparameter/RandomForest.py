import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

# Hàm tính Quadratic Kappa
def quadratic_kappa(y_true, y_pred):
    """Hàm tính Cohen Kappa Score với weights='quadratic'."""
    return cohen_kappa_score(y_true, np.round(y_pred), weights='quadratic')

# Đọc dữ liệu
data = pd.read_csv('/home/quoc/works/FinanceLearn/Data/Train_data_merge.csv')

# Xử lý dữ liệu
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

X = data.drop(columns=['sii'])  # Thay 'sii' bằng tên cột mục tiêu của bạn
y = data['sii']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hàm mục tiêu cho Optuna để tối ưu RandomForest
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),  # Removed 'auto'
        'random_state': 42
    }

    model = RandomForestRegressor(**params)
    
    # Tính Cross-Validation Score với Quadratic Kappa
    scorer = make_scorer(quadratic_kappa, greater_is_better=True)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring=scorer).mean()
    
    return score

# Tối ưu hóa với Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # Thử nghiệm với 50 lần

# Kết quả
print("Best parameters found for RandomForest:", study.best_params)
print("Best quadratic kappa score for RandomForest:", study.best_value)

# Đánh giá mô hình tốt nhất trên tập kiểm tra
best_params = study.best_params
best_model = RandomForestRegressor(**best_params)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
test_kappa_score = quadratic_kappa(y_test, y_pred)
print("Quadratic Kappa Score on test set (RandomForest):", test_kappa_score)
