import optuna
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import OneHotEncoder
import torch
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

# Hàm mục tiêu cho Optuna
def objective(trial):
    params = {
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_loguniform('lambda_sparse', 1e-5, 1e-3),
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': {'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2), 'weight_decay': 1e-5},
        'mask_type': trial.suggest_categorical('mask_type', ['entmax', 'sparsemax']),
        'scheduler_params': {'mode': "min", 'patience': 10, 'min_lr': 1e-5, 'factor': 0.5},
        'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'verbose': 0,
        'device_name': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Train TabNet
    model = TabNetRegressor(**params)
    model.fit(
        X_train=X_train.values, 
        y_train=y_train.values.reshape(-1, 1),
        eval_set=[(X_test.values, y_test.values.reshape(-1, 1))],
        eval_name=['valid'],
        eval_metric=['rmse'],
        max_epochs=200, 
        patience=20, 
        batch_size=1024, 
        virtual_batch_size=128, 
        drop_last=False
    )
    
    # Predict và tính Quadratic Kappa Score
    y_pred = model.predict(X_test.values).flatten()
    score = quadratic_kappa(y_test, y_pred)
    return score

# Tối ưu hóa với Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # Thử nghiệm với 150 lần

# Kết quả
print("Best parameters found for TabNet:", study.best_params)
print("Best quadratic kappa score for TabNet:", study.best_value)

# Đánh giá mô hình tốt nhất trên tập kiểm tra
best_params = study.best_params
best_model = TabNetRegressor(**best_params)
best_model.fit(
    X_train=X_train.values, 
    y_train=y_train.values.reshape(-1, 1),
    eval_set=[(X_test.values, y_test.values.reshape(-1, 1))],
    eval_name=['valid'],
    eval_metric=['rmse'],
    max_epochs=200, 
    patience=20, 
    batch_size=1024, 
    virtual_batch_size=128, 
    drop_last=False
)

y_pred = best_model.predict(X_test.values).flatten()
test_kappa_score = quadratic_kappa(y_test, y_pred)
print("Quadratic Kappa Score on test set (TabNet):", test_kappa_score)
