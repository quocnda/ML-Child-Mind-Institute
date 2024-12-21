from h2o.automl import H2OAutoML
import h2o
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import pandas as pd
# Khởi động H2O
print('Init')
h2o.init(verbose=True)
print('Ok')
data = h2o.import_file("/home/quoc/works/FinanceLearn/Data/Train_data_merge.csv")
train, test = data.split_frame(ratios=[.8])
print('Running .....')
# Chạy AutoML
aml = H2OAutoML(
    max_models=20,  # Số lượng mô hình tối đa
    seed=42,
    include_algos=["GBM", "XGBoost", "RandomForest"],  # Chỉ tập trung vào các thuật toán này
    sort_metric="logloss"  # Metric mặc định, sẽ tùy chỉnh sau bằng Cohen Kappa Score
)
aml.train(x=train.columns[:-1], y="sii", training_frame=train)

# Lấy leaderboard của AutoML
leaderboard = aml.leaderboard.as_data_frame()
print('Leaderboard:',leaderboard)
# Tính toán Cohen Kappa Score cho từng mô hình trong leaderboard
kappa_scores = []
for model_id in leaderboard['model_id']:
    model = h2o.get_model(model_id)
    predictions = model.predict(test)
    
    # Chuyển đổi nhãn dự đoán và nhãn thực tế
    y_pred = predictions['predict'].as_data_frame().values.flatten()
    y_true = test['sii'].as_data_frame().values.flatten()
    
    # Tính Cohen Kappa Score
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    kappa_scores.append((model_id, model, kappa))

# Sắp xếp các mô hình theo Cohen Kappa Score
sorted_kappa_scores = sorted(kappa_scores, key=lambda x: x[2], reverse=True)

# Lấy mô hình tốt nhất dựa trên Cohen Kappa Score
best_model_id = sorted_kappa_scores[0][0]
best_model = sorted_kappa_scores[0][1]
best_kappa_score = sorted_kappa_scores[0][2]

# In kết quả
print(f"Best model based on Cohen Kappa Score: {best_model_id}")
print(f"Cohen Kappa Score: {best_kappa_score}")

# In các hyperparameter của mô hình tốt nhất
print("Best model hyperparameters:")
for param, value in best_model.params.items():
    print(f"{param}: {value['actual']}")
