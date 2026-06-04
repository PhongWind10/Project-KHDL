from matplotlib.lines import Line2D      # Tạo các đối tượng đường (Line2D) dùng trong custom legend, annotation trên biểu đồ
import pandas as pd                      # Xử lý và phân tích dữ liệu dạng bảng (DataFrame), đọc/ghi file CSV, thao tác dữ liệu 
import numpy as np                       # Toán học và xử lý mảng (ndarray), tính toán số học hiệu quả trên dữ liệu lớn 
import matplotlib.pyplot as plt          # Vẽ biểu đồ 2D (line, scatter, bar, histogram…) trên nền Matplotlib
import seaborn as sns                   # Thư viện trực quan hóa nâng cao dựa trên Matplotlib, dễ vẽ các biểu đồ thống kê 
from sklearn.linear_model import LogisticRegression          # Mô hình Hồi quy Logistic dùng sẵn trong scikit-learn 
from sklearn.model_selection import train_test_split, cross_val_score  # Chia tập dữ liệu và đánh giá chéo (cross-validation) 
from sklearn.preprocessing import StandardScaler             # Chuẩn hóa (Standardization) các feature về phân phối chuẩn 
from sklearn.metrics import (
    accuracy_score,        # Tính độ chính xác (Accuracy)
    auc,                   # Tính diện tích dưới đường cong ROC
    confusion_matrix,      # Ma trận nhầm lẫn
    classification_report,
    precision_recall_curve, # Báo cáo precision/recall/F1
    roc_curve,             # Lấy tọa độ ROC để vẽ đồ thị
    roc_auc_score         # Tính AUC cho ROC
) 
from sklearn.metrics import (
    precision_recall_curve,    # Tính các cặp (precision, recall) tương ứng với nhiều ngưỡng dự báo khác nhau trong bài toán phân loại nhị phân 
    average_precision_score    # Tính chỉ số Average Precision (AP): trung bình có trọng số của các giá trị precision tại mỗi thay đổi recall 
)
from sklearn.tree import DecisionTreeClassifier  # Mô hình Cây quyết định dùng sẵn trong scikit-learn 
from imblearn.over_sampling import SMOTE         # Kỹ thuật SMOTE để tăng cường mẫu cho lớp thiểu số 
from statsmodels.nonparametric.smoothers_lowess import lowess  # Hàm LOWESS (Locally Weighted Scatterplot Smoothing) để làm mượt đường xu hướng 
# Thử import pydotplus nếu có, nếu không có sẽ dùng fallback với sklearn.tree.plot_tree để xuất ảnh
try:
    import pydotplus               # Tạo và xuất file DOT (Graphviz) để trực quan hóa cây quyết định 
    HAS_PYDOTPLUS = True
except Exception:
    pydotplus = None
    HAS_PYDOTPLUS = False

from sklearn import tree       # Module con của scikit-learn hỗ trợ xuất thành DOT và vẽ cây 
from matplotlib.colors import ListedColormap  # Tạo colormap tùy chỉnh cho đồ thị phân lớp (decision regions) 
from sklearn.calibration import calibration_curve  # Tính điểm hiệu chỉnh (calibration curve) để đánh giá khả năng ước lượng xác suất của mô hình 

#++++++++++++++++++PHẦN I: DỰ ĐOÁN MÔ HÌNH HỒI QUY LOGISTIC VÀ CÂY QUYẾT ĐỊNH+++++++++++++++++
# 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
file_path = 'D:/accident.csv'
df = pd.read_csv(file_path, sep=';')  # Đọc file CSV, ngăn cách bởi dấu chấm phẩy

# Xử lý giá trị thiếu:
df['Speed_of_Impact'].fillna(df['Speed_of_Impact'].mean(), inplace=True)  # Thay thế bằng giá trị trung bình
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)  # Thay thế bằng mode (giá trị phổ biến nhất)

# Mã hóa biến phân loại thành dạng nhị phân:
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
binary_columns = ['Helmet_Used', 'Seatbelt_Used']
for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Lưu dữ liệu sau xử lý ra file
df.to_csv('D:/accident_preprocessed.csv', index=False, sep=';')
print("\nDữ liệu sau tiền xử lý đã được lưu vào file: D:/accident_preprocessed.csv")

# 2. CHUẨN BỊ DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH HỒI QUY LOGISTIC
features = ['Age', 'Gender', 'Speed_of_Impact', 'Helmet_Used', 'Seatbelt_Used']
target = 'Survived'
X = df[features]
y = df[target]

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cân bằng dữ liệu bằng SMOTE (tăng cường mẫu thiểu số)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Chia dữ liệu thành 3 phần: train 60%, validation 20%, test 20%
X_temp, X_test, y_temp, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Hàm sigmoid dùng trong Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hàm tính hàm mất mát log loss
def compute_loss(y, y_pred):
    m = len(y)
    return - (1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Huấn luyện Logistic Regression bằng gradient descent thủ công
def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    for _ in range(epochs):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)
        weights -= learning_rate * dw
        bias -= learning_rate * db
        if _ % 100 == 0:
            loss = compute_loss(y, y_pred)
            print(f"Epoch {_}, Loss: {loss:.4f}")
    return weights, bias

# Huấn luyện mô hình Logistic Regression
weights, bias = logistic_regression(X_train, y_train)

# Dự đoán xác suất và nhị phân trên tập test
y_pred_proba = sigmoid(np.dot(X_test, weights) + bias)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Đánh giá mô hình Logistic Regression
accuracy = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác của mô hình: {accuracy:.2f}")
print("\nMa trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred))
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred))

# Cross-validation với Logistic Regression từ thư viện sklearn
logreg_sklearn = LogisticRegression()
scores = cross_val_score(logreg_sklearn, X_resampled, y_resampled, cv=5, scoring='accuracy')
print("\nĐánh giá Cross-validation (Hồi quy Logistic - sklearn):")
print("Độ chính xác trung bình:", round(np.mean(scores), 2))

# Xuất kết quả dự đoán Logistic Regression ra CSV
results = pd.DataFrame({
    'Age': X_test[:, 0],
    'Gender': X_test[:, 1],
    'Speed_of_Impact': X_test[:, 2],
    'Helmet_Used': X_test[:, 3],
    'Seatbelt_Used': X_test[:, 4],
    'Actual_Survived': y_test,
    'Predicted_Survived': y_pred
})
results.to_csv('D:/accident_logistic_regression.csv', index=False, sep=';')
print("\nKết quả dự đoán đã được lưu vào file: accident_logistic_regression.csv")

# 5. HUẤN LUYỆN VÀ ĐÁNH GIÁ CÂY QUYẾT ĐỊNH
# Khởi tạo mô hình Cây quyết định với các tham số tối ưu
dt_model = DecisionTreeClassifier(
    max_depth=6,                # Độ sâu tối đa của cây
    min_samples_split=4,        # Số lượng mẫu tối thiểu để chia nút
    min_samples_leaf=2,         # Số mẫu tối thiểu trong mỗi lá
    criterion='gini',           # Chỉ số dùng để chia (Gini)
    max_features=None,
    random_state=42
)

# Huấn luyện và dự đoán
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Đánh giá mô hình Decision Tree
print(f"\n[Decision Tree] Độ chính xác: {accuracy_score(y_test, y_pred_dt):.2f}")
print("\n[Decision Tree] Ma trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred_dt))
print("\n[Decision Tree] Báo cáo phân loại:")
print(classification_report(y_test, y_pred_dt))

# Lưu kết quả Decision Tree ra file
df_output = pd.DataFrame(X_test, columns=features)
df_output['Actual'] = y_test
df_output['Predicted'] = y_pred_dt
df_output.to_csv('accident_decision_tree_results.csv', index=False)
print("\nKết quả dự đoán của mô hình cây quyết định đã được lưu vào file: accident_decision_tree_results.csv")

#++++++++++++++++++++PHẦN II: VẼ CÁC BIỂU ĐỒ HỒI QUY LOGISTIC VÀ CÂY QUYẾT ĐỊNH ĐỂ SO SÁNH+++++++++++++++++++
# 1. Biểu đồ phân tán mô hình Hồi Quy Logistic chỉ với 2 đặc trưng (Age và Speed_of_Impact)
feature_indices = [0, 2]  # Chỉ chọn hai cột: Age và Speed_of_Impact
X_vis_train = X_train[:, feature_indices]  # Tập huấn luyện chỉ có 2 đặc trưng
X_vis_test = X_test[:, feature_indices]    # Tập kiểm thử chỉ có 2 đặc trưng

# --- Huấn luyện lại Logistic Regression với 2 đặc trưng đã chọn ---
log_reg_vis = LogisticRegression()
log_reg_vis.fit(X_vis_train, y_train)  # Học mô hình trên tập train

# --- Tạo lưới điểm (meshgrid) cho không gian 2 chiều của tập train ---
grid_x, grid_y = np.meshgrid(
    np.linspace(X_vis_train[:, 0].min() - 1,
                X_vis_train[:, 0].max() + 1, 300),  # Phạm vi Age
    np.linspace(X_vis_train[:, 1].min() - 1,
                X_vis_train[:, 1].max() + 1, 300)   # Phạm vi Speed_of_Impact
)
grid = np.c_[grid_x.ravel(), grid_y.ravel()]  # Ghép thành tọa độ 2 chiều
probs_train = log_reg_vis.predict_proba(grid)[:, 1].reshape(grid_x.shape)  # Xác suất sống sót

# --- Tạo lưới điểm cho tập test ---
xx, yy = np.meshgrid(
    np.linspace(X_vis_test[:, 0].min() - 1,
                X_vis_test[:, 0].max() + 1, 300),
    np.linspace(X_vis_test[:, 1].min() - 1,
                X_vis_test[:, 1].max() + 1, 300)
)
grid_test = np.c_[xx.ravel(), yy.ravel()]
probs_test = log_reg_vis.predict_proba(grid_test)[:, 1].reshape(xx.shape)  # Xác suất trên tập test

# --- Dự đoán nhãn trên tập Test chỉ với 2 đặc trưng ---
y_pred_vis = log_reg_vis.predict(X_vis_test)

# --- Vẽ biểu đồ phân tán và vùng quyết định ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1a) Biểu đồ trên tập Train với vùng màu thể hiện xác suất >0.5
axes[0].contourf(grid_x, grid_y, probs_train,
                 levels=[0, 0.5, 1], cmap="RdYlGn", alpha=0.6)  # Vùng quyết định
for label, color, name in zip([0, 1], ['red', 'green'], ['Không sống sót', 'Sống sót']):
    idx = (y_train == label)
    axes[0].scatter(X_vis_train[idx, 0], X_vis_train[idx, 1],
                    c=color, label=name, edgecolors='k', alpha=0.7)  # Dữ liệu thực tế
axes[0].set(title="Biểu đồ logistic trên tập Train (Thực tế)",
            xlabel="Age (chuẩn hóa)", ylabel="Speed_of_Impact (chuẩn hóa)")
axes[0].legend(title='Survived')
axes[0].grid(True)

# 1b) Biểu đồ trên tập Test với kết quả dự đoán làm nhãn điểm
axes[1].contourf(xx, yy, probs_test,
                 levels=[0, 0.5, 1], cmap="RdYlGn", alpha=0.6)
for label, color, name in zip([0, 1], ['red', 'green'], ['Không sống sót', 'Sống sót']):
    idx = (y_pred_vis == label)
    axes[1].scatter(X_vis_test[idx, 0], X_vis_test[idx, 1],
                    c=color, label=name, edgecolors='k', alpha=0.7)  # Dữ liệu dự đoán
axes[1].set(title="Biểu đồ logistic trên tập Test (Dự đoán)",
            xlabel="Age (chuẩn hóa)", ylabel="Speed_of_Impact (chuẩn hóa)")
axes[1].legend(title='Survived')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 2. Biểu đồ đường Sigmoid trên tổng trọng số đầu vào của mô hình
# Tính tổng trọng số: s = w^T x + b cho train và test
s_train = np.dot(X_train, weights) + bias  # Tổng trọng số của mỗi mẫu train
s_test  = np.dot(X_test,  weights) + bias  # Tổng trọng số của mỗi mẫu test

# Tạo figure với 2 axes để so sánh
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 2a) Sigmoid trên dữ liệu thực tế (train)
for label, color, name, y_value in zip(
    [0, 1], ['green', 'red'],
    ['No: Không sống sót (0)', 'Yes: Có sống sót (1)'],
    [0.02, 0.98]
):
    idx = (y_train == label)
    axes[0].scatter(
        s_train[idx], 
        [y_value] * idx.sum(),
        c=color, marker='o', edgecolors='k',
        label=name, alpha=0.8
    )  # Điểm dữ liệu theo nhãn thực tế
# Vẽ đường sigmoid lý thuyết
x_vals = np.linspace(-4, 4, 200)
axes[0].plot(x_vals, sigmoid(x_vals), label='Sigmoid', linestyle='-', linewidth=2)
axes[0].axhline(0.5, color='black', linestyle='--', label='Ngưỡng 0.5')  # Ngưỡng phân lớp
axes[0].set(
    title="Biểu đồ đường Sigmoid trên dữ liệu thực tế (Train)",
    xlabel="Tổng trọng số đầu vào (s)",
    ylabel="Xác suất sigmoid(s)"
)
axes[0].legend()
axes[0].grid(True)

# 2b) Sigmoid trên dữ liệu dự đoán (test) – tương tự train
for label, color, name, y_value in zip(
    [0, 1], ['green', 'red'],
    ['No: Không sống sót (0)', 'Yes: Có sống sót (1)'],
    [0.02, 0.98]
):
    idx = (y_pred == label)
    axes[1].scatter(
        s_test[idx],
        [y_value] * idx.sum(),
        c=color, marker='o', edgecolors='k',
        label=name, alpha=0.8
    )  # Điểm dự đoán của mô hình
axes[1].plot(x_vals, sigmoid(x_vals), label='Sigmoid', linestyle='-', linewidth=2)
axes[1].axhline(0.5, color='black', linestyle='--', label='Ngưỡng 0.5')
axes[1].set(
    title="Biểu đồ đường Sigmoid trên dữ liệu dự đoán (Test)",
    xlabel="Tổng trọng số đầu vào (s)",
    ylabel="Xác suất sigmoid(s)"
)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 3. Biểu đồ Calibration Curve (Đường hiệu chỉnh xác suất)
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', label='Đường hiệu chỉnh hoàn hảo')  # Tuyến đường chuẩn
plt.plot(mean_predicted_value, fraction_of_positives, marker='.', label='Mô hình Logistic')  # Hiệu chỉnh thực tế
plt.xlabel('Xác suất dự đoán trung bình')
plt.ylabel('Tỷ lệ thực tế của kết quả dương')
plt.title('Biểu đồ đường hiệu chỉnh (Calibration Curve)')
plt.legend()
plt.grid(True)
plt.show()

# 4. Biểu đồ Residuals (Phần dư giữa nhãn thật và xác suất dự đoán)
residuals = y_test - y_pred_proba  # Sai số dự đoán
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_proba, residuals)
plt.xlabel('Xác suất dự đoán')
plt.ylabel('Phần dư')
plt.title('Biểu đồ phần dư')
plt.axhline(y=0, color='r', linestyle='--')  # Trục tham chiếu
plt.grid(True)
plt.show()

#5. Biểu đồ Phân phối xác suất dự đoán sống sót (thực tế và dự đoán)
# 5a) Phân phối độ tuổi: Thực tế vs Dự đoán
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 5a.1) Phân phối thực tế
sns.histplot(df[df['Survived'] == 1]['Age'], kde=True,
             color='green', ax=axes[0], label='Thực tế: Sống sót', alpha=0.5)
sns.histplot(df[df['Survived'] == 0]['Age'], kde=True,
             color='red', ax=axes[0], label='Thực tế: Không sống sót', alpha=0.5)
axes[0].set(title='Phân phối xác suất độ tuổi theo dữ liệu Thực tế', xlabel='Tuổi', ylabel='Tần suất')
axes[0].legend()

# 5a.2) Phân phối dự đoán
age_test = scaler.inverse_transform(X_test)[:, 0]
sns.histplot(age_test[y_pred == 1], kde=True,
             color='green', ax=axes[1], label='Dự đoán: Sống sót', alpha=0.5)
sns.histplot(age_test[y_pred == 0], kde=True,
             color='red', ax=axes[1], label='Dự đoán: Không sống sót', alpha=0.5)
axes[1].set(title='Phân phối xác suất độ tuổi theo kết quả Dự đoán', xlabel='Tuổi', ylabel='Tần suất')
axes[1].legend()
plt.tight_layout()
plt.show()

# 5b) Phân phối tốc độ va chạm: Thực tế vs Dự đoán
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 5b.1) Phân phối thực tế
sns.histplot(df[df['Survived'] == 1]['Speed_of_Impact'], kde=True,
             color='green', ax=axes[0], label='Thực tế: Sống sót', alpha=0.5)
sns.histplot(df[df['Survived'] == 0]['Speed_of_Impact'], kde=True,
             color='red', ax=axes[0], label='Thực tế: Không sống sót', alpha=0.5)
axes[0].set(title='Phân phối xác suất tốc độ va chạm theo dữ liệu Thực tế', xlabel='Tốc độ va chạm', ylabel='Tần suất')
axes[0].legend()
# 5b.2) Phân phối dự đoán
speed_test = scaler.inverse_transform(X_test)[:, 2]
sns.histplot(speed_test[y_pred == 1], kde=True,
             color='green', ax=axes[1], label='Dự đoán: Sống sót', alpha=0.5)
sns.histplot(speed_test[y_pred == 0], kde=True,
             color='red', ax=axes[1], label='Dự đoán: Không sống sót', alpha=0.5)
axes[1].set(title='Phân phối xác suất tốc độ va chạm theo kết quả Dự đoán', xlabel='Tốc độ va chạm', ylabel='Tần suất')
axes[1].legend()
plt.tight_layout()
plt.show()

# 5c) Biểu đồ cột nhóm giới tính: Thực tế vs Dự đoán
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 5c.1) Biểu đồ cột nhóm thực tế
sns.countplot(x='Gender', hue='Survived', data=df, ax=axes[0],
              palette={1:'green', 0:'red'})
axes[0].set(title='So sánh số lượng giới tính sống sót và \nkhông sống sót giữa dữ liệu Thực tế', xlabel='Giới tính (0: Nữ, 1: Nam)', ylabel='Số lượng')
axes[0].legend(title='Thực tế', labels=['Không sống sót', 'Sống sót'])
# 5c.2) Phân phối dự đoán
pred_df = pd.DataFrame(scaler.inverse_transform(X_test), columns=features)
pred_df['Dự đoán'] = y_pred
sns.countplot(x='Gender', hue='Dự đoán', data=pred_df, ax=axes[1],
              palette={1:'green', 0:'red'})
axes[1].set(title='So sánh số lượng giới tính sống sót và \nkhông sống sót theo kết quả Dự đoán', xlabel='Giới tính (0: Nữ, 1: Nam)', ylabel='Số lượng')
axes[1].legend(title='Dự đoán', labels=['Không sống sót', 'Sống sót'])
plt.tight_layout()
plt.show()

# 5d) Phân phối việc sử dụng mũ bảo hiểm: Thực tế vs Dự đoán
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 5d.1) Phân phối thực tế
sns.countplot(x='Helmet_Used', hue='Survived', data=df, ax=axes[0],
              palette={1:'green', 0:'red'})
axes[0].set(title='So sánh số lượng sử dụng mũ bảo hiểm, sống sót và \nkhông sống sót giữa dữ liệu Thực tế', xlabel='Mũ bảo hiểm (0: Không, 1: Có)', ylabel='Số lượng')
axes[0].legend(title='Thực tế', labels=['Không sống sót', 'Sống sót'])
# 5d.2) Phân phối dự đoán
sns.countplot(x='Helmet_Used', hue='Dự đoán', data=pred_df, ax=axes[1],
              palette={1:'green', 0:'red'})
axes[1].set(title='So sánh số lượng sử dụng mũ bảo hiểm, sống sót và \nkhông sống sót theo kết quả Dự đoán', xlabel='Mũ bảo hiểm (0: Không, 1: Có)', ylabel='Số lượng')
axes[1].legend(title='Dự đoán', labels=['Không sống sót', 'Sống sót'])
plt.tight_layout()
plt.show()

# 5e) Phân phối việc sử dụng dây an toàn: Thực tế vs Dự đoán
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 5e.1) Phân phối thực tế
sns.countplot(x='Seatbelt_Used', hue='Survived', data=df, ax=axes[0],
              palette={1:'green', 0:'red'})
axes[0].set(title='So sánh số lượng sử dụng dây an toàn, sống sót và \nkhông sống sót giữa dữ liệu Thực tế', xlabel='Dây an toàn (0: Không, 1: Có)', ylabel='Số lượng')
axes[0].legend(title='Thực tế', labels=['Không sống sót', 'Sống sót'])
# 5e.2) Phân phối dự đoán
sns.countplot(x='Seatbelt_Used', hue='Dự đoán', data=pred_df, ax=axes[1],
              palette={1:'green', 0:'red'})
axes[1].set(title='So sánh số lượng sử dụng dây an toàn, sống sót và \nkhông sống sót theo kết quả Dự đoán', xlabel='Dây an toàn (0: Không, 1: Có)', ylabel='Số lượng')
axes[1].legend(title='Dự đoán', labels=['Không sống sót', 'Sống sót'])
plt.tight_layout()
plt.show()

# 5f) Phân phối xác suất dự đoán sống sót
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(y_pred_proba[y_pred == 1], kde=True, color='green',
             ax=axes[0], label='Dự đoán sống sót')
axes[0].set(title='Phân phối xác suất dự đoán khả năng có sống sót', #Dự đoán sống sót
            xlabel='Xác suất', ylabel='Tần suất')
axes[0].legend()
axes[0].grid(True)

sns.histplot(y_pred_proba[y_pred == 0], kde=True, color='red',
             ax=axes[1], label='Dự đoán không sống sót')
axes[1].set(title='Phân phối xác suất dự đoán khả năng không sống sót', #Dự đoán không sống sót
            xlabel='Xác suất', ylabel='Tần suất')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 6. Biểu đồ Ma trận tương quan giữa các biến
correlation_matrix = df[features + [target]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', fmt=".2f") #Hiển thị hệ số tương quan
plt.title('Ma trận tương quan giữa các biến')
plt.show()

# 7. Ma trận nhầm lẫn của Logistic Regression và Decision Tree
cm_logistic = confusion_matrix(y_test, y_pred)      # Ma trận cho Logistic
cm_dt       = confusion_matrix(y_test, y_pred_dt)  # Ma trận cho Decision Tree
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Vẽ heatmap cho từng ma trận
sns.heatmap(cm_logistic, annot=True, fmt='d', ax=axes[0], cmap='RdYlGn',
            xticklabels=['Không sống sót', 'Sống sót'], yticklabels=['Không sống sót', 'Sống sót'])
axes[0].set(title='Confusion Matrix - Logistic', xlabel='Dự đoán', ylabel='Thực tế')
sns.heatmap(cm_dt, annot=True, fmt='d', ax=axes[1], cmap='RdYlGn',
            xticklabels=['Không sống sót', 'Sống sót'], yticklabels=['Không sống sót', 'Sống sót'])
axes[1].set(title='Confusion Matrix - Decision Tree', xlabel='Dự đoán', ylabel='Thực tế')
plt.tight_layout()
plt.show()

# 8. Vẽ biểu đồ ROC Curve và tính AUC cho mô hình hồi quy logistic và cây quyết định
# 8a. Tính xác suất nhãn 1 và AUC cho Logistic Regression
y_pred_proba_lr = dt_model.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
roc_auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
# 8b. Tính xác suất nhãn 1 và AUC cho Decision Tree
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
roc_auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
# 8c. Vẽ ROC Curve của cả hai mô hình
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr,
         label=f"Logistic Regression (AUC = {roc_auc_lr:.2f})",
         linestyle='-', linewidth=2)
plt.plot(fpr_dt, tpr_dt,
         label=f"Decision Tree (AUC = {roc_auc_dt:.2f})",
         linestyle='--', linewidth=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
plt.xlabel('Tỷ lệ dương tính giả (FPR)')
plt.ylabel('Tỷ lệ dương tính đúng (TPR)')
plt.title('Đường ROC cho Hồi quy Logistic và Cây quyết định')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 9) Đường cong Precision-Recall
# 9a.1) Logistic Regression
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_proba)
ap_lr = average_precision_score(y_test, y_pred_proba)
# 9a.2) Decision Tree
precision_dt, recall_dt, _ = precision_recall_curve(y_test, y_pred_proba_dt)
ap_dt = average_precision_score(y_test, y_pred_proba_dt)
plt.figure(figsize=(8,6))
plt.plot(recall_lr, precision_lr, label=f"Logistic (AP = {ap_lr:.2f})", linestyle='-', linewidth=2)
plt.plot(recall_dt, precision_dt, label=f"Decision Tree (AP = {ap_dt:.2f})", linestyle='--', linewidth=2)
plt.xlabel('Độ thu hồi (Recall)')
plt.ylabel('Độ chính xác (Precision)')
plt.title('Đường cong Precision-Recall')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

# 10) Biểu đồ Tỷ lệ Chấp Thuận (Odds Ratio)
# Lấy hệ số từ mô hình sklearn LogisticRegression
theta = logreg_sklearn.fit(X_resampled, y_resampled).coef_[0]
odds_ratio = np.exp(theta)
features_names = features
plt.figure(figsize=(8,6))
plt.barh(features_names, odds_ratio)
plt.xlabel('Tỷ lệ chấp thuận (Odds Ratio)')
plt.title('Biểu đồ Tỷ lệ Chấp Thuận (Odds Ratio) - Logistic Regression')
plt.grid(axis='x')
plt.show()

# 11) Biểu đồ Lift Chart
from sklearn.metrics import precision_recall_curve
# Sắp xếp xác suất và nhãn thực tế
data_lr = pd.DataFrame({'y_true': y_test, 'y_prob': y_pred_proba}).sort_values('y_prob', ascending=False)
# Tính lift theo bội lượng
total_pos = data_lr['y_true'].sum()
data_lr['cum_pos'] = data_lr['y_true'].cumsum()
data_lr['cum_pct_pos'] = data_lr['cum_pos'] / total_pos
data_lr['cum_pct_obs'] = np.arange(1, len(data_lr)+1) / len(data_lr)
plt.figure(figsize=(8,6))
plt.plot(data_lr['cum_pct_obs'], data_lr['cum_pct_pos'], label='Logistic', linewidth=2)
plt.plot(data_lr['cum_pct_obs'], data_lr['cum_pct_obs'], linestyle='--', label='Ngẫu nhiên')
plt.xlabel('Tỷ lệ mẫu (%)')
plt.ylabel('Tỷ lệ dương tính tích lũy (%)')
plt.title('Biểu đồ Lift Chart - Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

# 12) Biểu đồ Log Odds vs X cho từng biến (chỉ Logistic)
for i, feat in enumerate(features):
    plt.figure(figsize=(8,6))
    plt.scatter(scaler.inverse_transform(X_test)[:, i],
                np.log((y_pred_proba + 1e-6)/(1 - y_pred_proba - 1e-6)), alpha=0.6)
    plt.xlabel(feat)
    plt.ylabel('Log Odds')
    plt.title(f'Biểu đồ Log Odds vs {feat}')
    plt.grid(True)
    plt.show()

# 13) Biểu đồ Tầm quan trọng Đặc trưng (Feature Importance)
importances = dt_model.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(8,6))
plt.barh([features[i] for i in indices], importances[indices])
plt.xlabel('Độ quan trọng')
plt.title('Biểu đồ Tầm quan trọng Đặc trưng - Decision Tree')
plt.grid(axis='x')
plt.show()

# 14) Phân phối Xác suất của phân phối Logistic (Logistic Distribution)
from scipy.stats import logistic
x_ld = np.linspace(0,1,200)
plt.figure(figsize=(8,6))
plt.plot(x_ld, logistic.pdf(x_ld), linewidth=2)
plt.xlabel('Xác suất')
plt.ylabel('Mật độ xác suất')
plt.title('Phân phối Xác suất (Logistic Distribution)')
plt.grid(True)
plt.show()

# 15. Xuất cây quyết định ra ảnh PNG
# Thử dùng pydotplus nếu có, nếu không có dùng sklearn.tree.plot_tree + matplotlib làm fallback
dot_data = tree.export_graphviz(
    dt_model,
    out_file=None,
    feature_names=features,
    class_names=['Không sống sót', 'Sống sót'],
    filled=True,
    rounded=True,
    special_characters=True
)
try:
    if HAS_PYDOTPLUS and pydotplus is not None:
        graph = pydotplus.graph_from_dot_data(dot_data)  # Tạo đối tượng đồ họa
        graph.write_png("decision_tree.png")  # Lưu thành file ảnh
        print("Cây quyết định đã được lưu tại: decision_tree.png (via pydotplus)")
    else:
        # Fallback: vẽ trực tiếp bằng sklearn và matplotlib
        plt.figure(figsize=(20, 10))
        tree.plot_tree(
            dt_model,
            feature_names=features,
            class_names=['Không sống sót', 'Sống sót'],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.tight_layout()
        plt.savefig("decision_tree.png", bbox_inches='tight')
        plt.close()
        print("Cây quyết định đã được lưu tại: decision_tree.png (via sklearn.plot_tree)")
except Exception as e:
    print("Không thể xuất cây quyết định:", e)