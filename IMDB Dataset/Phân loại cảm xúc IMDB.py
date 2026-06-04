# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH PHÂN TÍCH CẢM XÚC IMDB SỬ DỤNG MÔ HÌNH LAI CNN - Bi-LSTM
(PHIÊN BẢN CHẠY TRÊN VS CODE / LOCAL MÁY TÍNH)

HƯỚNG DẪN CÀI ĐẶT MÔI TRƯỜNG:
1. Mở Terminal trong VS Code.
2. Cài đặt các thư viện cần thiết bằng lệnh:
   pip install pandas numpy matplotlib seaborn nltk tensorflow scikit-learn

3. Đặt file 'IMDB_Dataset.csv' vào CÙNG THƯ MỤC với file code này.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
# Try to import NLTK; if unavailable, fall back to a small stopwords set
try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except Exception:
    nltk = None
    NLTK_AVAILABLE = False
    # Minimal fallback stopwords list to allow preprocessing without NLTK
    fallback_stopwords = {
        'a','an','the','and','or','if','in','on','at','for','to','is','are','was','were','be','been','being',
        'of','with','as','by','that','this','these','those','it','its','from','but','not','they','their','them',
        'he','she','his','her','you','your','i','we','our','us'
    }
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.decomposition import PCA

# TensorFlow Imports
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Conv1D, MaxPooling1D, 
                                     Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG VÀ THAM SỐ
# ==========================================

# Kiểm tra GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Đã phát hiện GPU: {gpus}. Quá trình train sẽ nhanh hơn.")
else:
    print("Không tìm thấy GPU. Sử dụng CPU (có thể sẽ chậm hơn).")

# Cấu hình tham số mô hình
VOCAB_SIZE = 10000    # Số lượng từ vựng
EMBEDDING_DIM = 100   # Kích thước vector
MAX_LENGTH = 500      # Độ dài chuỗi
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = "<OOV>"
EPOCHS = 5            # Số epoch
BATCH_SIZE = 64
DATA_FILE = 'D:\IMDB_Dataset.csv' # Tên file dữ liệu

# Tải dữ liệu NLTK (chỉ chạy 1 lần nếu chưa có)
if NLTK_AVAILABLE:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Đang tải dữ liệu NLTK Stopwords...")
        nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))
else:
    print("NLTK không khả dụng; sử dụng danh sách stopwords thay thế.")
    STOPWORDS = fallback_stopwords

# ==========================================
# 2. HÀM TẢI VÀ XỬ LÝ DỮ LIỆU
# ==========================================

def load_local_data(filename):
    """
    Hàm đọc dữ liệu từ file csv nằm cùng thư mục.
    """
    if not os.path.exists(filename):
        print(f"LỖI: Không tìm thấy file '{filename}' trong thư mục hiện tại.")
        print(f"Thư mục hiện tại đang là: {os.getcwd()}")
        return None
    
    print(f"Đang đọc dữ liệu từ: {filename}...")
    try:
        df = pd.read_csv(filename)
        print(f"Đọc thành công {len(df)} dòng dữ liệu.")
        return df
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return None

def clean_text(text):
    """Làm sạch văn bản cơ bản"""
    # Xóa HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Xóa ký tự đặc biệt, giữ lại chữ cái
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Chuyển thường và tách từ
    text = text.lower().split()
    # Xóa stopwords
    text = [word for word in text if word not in STOPWORDS]
    return ' '.join(text)

def preprocess_data(df):
    print("Đang tiền xử lý dữ liệu (Clean, Tokenize, Padding)...")
    
    # 1. Clean Text
    # Sử dụng .progress_apply nếu muốn hiện thanh tiến trình (cần cài tqdm), ở đây dùng apply thường
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # 2. Encode Label
    df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    # Chia tập dữ liệu
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['cleaned_review'], df['label'], test_size=0.2, random_state=42
    )
    
    # 3. Tokenize
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
    tokenizer.fit_on_texts(X_train_text)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)
    
    # 4. Pad Sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer

# ==========================================
# 3. XÂY DỰNG MÔ HÌNH CNN-BiLSTM
# ==========================================

def build_cnn_bilstm_model(vocab_size, embedding_dim, input_length):
    model = Sequential([
        # Embedding Layer
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        SpatialDropout1D(0.2),
        
        # CNN Layer (Feature Extraction)
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        
        # Bi-LSTM Layer (Sequence/Context Learning)
        Bidirectional(LSTM(64, return_sequences=False)),
        
        # Dense Layers
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(learning_rate=0.001), 
                  metrics=['accuracy'])
    return model

# ==========================================
# 4. TRỰC QUAN HÓA VÀ ĐÁNH GIÁ
# ==========================================

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    print("Đang hiển thị biểu đồ Training History. Vui lòng đóng cửa sổ biểu đồ để tiếp tục...")
    plt.show()

def evaluate_model(model, X_test, y_test):
    print("\nĐang dự đoán trên tập Test...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = y_test.values
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Hiển thị bảng chỉ số
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'Value': [accuracy, precision, recall, f1, roc_auc]
    })
    
    print("\n" + "="*40)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("="*40)
    print(metrics_df)
    print("="*40 + "\n")
    
    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    print("Đang hiển thị Confusion Matrix...")
    plt.show()
    
    # ROC Curve Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    print("Đang hiển thị ROC Curve...")
    plt.show()

def visualize_embeddings(model, tokenizer, num_words=200):
    """Vẽ biểu đồ vector từ"""
    try:
        embedding_layer = model.layers[0]
        weights = embedding_layer.get_weights()[0]
        
        word_index = tokenizer.word_index
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        
        words_to_plot = []
        vectors_to_plot = []
        
        for i in range(1, num_words + 1):
            if i in reverse_word_index:
                word = reverse_word_index[i]
                words_to_plot.append(word)
                vectors_to_plot.append(weights[i])
                
        pca = PCA(n_components=2)
        result = pca.fit_transform(vectors_to_plot)
        
        plt.figure(figsize=(12, 10))
        plt.scatter(result[:, 0], result[:, 1])
        
        for i, word in enumerate(words_to_plot):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=9)
            
        plt.title(f'Word Embeddings Visualization (Top {num_words})')
        plt.grid(True)
        print("Đang hiển thị biểu đồ Embedding...")
        plt.show()
    except Exception as e:
        print(f"Không thể vẽ Embedding (có thể do lỗi PCA hoặc dữ liệu): {e}")

# ==========================================
# 5. CHƯƠNG TRÌNH CHÍNH
# ==========================================

if __name__ == "__main__":
    # Load Data
    df = load_local_data(DATA_FILE)
    
    if df is not None:
        # Preprocess
        X_train, X_test, y_train, y_test, tokenizer = preprocess_data(df)
        print(f"Kích thước tập Train: {X_train.shape}")
        print(f"Kích thước tập Test: {X_test.shape}")
        
        # Build Model
        model = build_cnn_bilstm_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH)
        model.summary()
        
        # Train
        print("\nBắt đầu huấn luyện mô hình...")
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = model.fit(X_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stop],
                            verbose=1)
        
        # Plot & Evaluate
        plot_training_history(history)
        evaluate_model(model, X_test, y_test)
        visualize_embeddings(model, tokenizer)
        
        print("\nHoàn tất chương trình.")
    else:
        print("\nDừng chương trình do không tải được dữ liệu.")