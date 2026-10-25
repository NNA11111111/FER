import pandas as pd
import numpy as np
import cv2

def load_fer2013_csv(img_csv_path, label_csv_path):
    print("Đang ghép dữ liệu ảnh và nhãn FER+ (8 lớp)...")
    
    # Đọc file cũ lấy ảnh và file mới lấy nhãn
    df_img = pd.read_csv(img_csv_path)
    df_label = pd.read_csv(label_csv_path)
    
    # Ghép cột pixels từ df_img sang df_label (vì 2 file này khớp nhau từng dòng)
    df_label['pixels'] = df_img['pixels']
    
    # Lọc trùng lặp
    df_label = df_label.drop_duplicates(subset=['pixels'])
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    
    # Các cột nhãn của FER+ theo thứ tự 0->7
    emotion_cols = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    
    for _, row in df_label.iterrows():
        usage = row['Usage']
        
        # FER+ dùng phương pháp voting, ta lấy cảm xúc có số phiếu bầu cao nhất làm nhãn chính
        votes = row[emotion_cols].values.astype(int)
        emotion = np.argmax(votes) 
        
        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8).reshape(48, 48)
        img_clahe = clahe.apply(pixels)
        img_norm = (img_clahe.astype(np.float32) / 255.0 - 0.5) / 0.5
        img_final = np.expand_dims(img_norm, axis=-1)
        
        if usage == 'Training':
            X_train.append(img_final)
            y_train.append(emotion)
        elif usage == 'PublicTest':
            X_val.append(img_final)
            y_val.append(emotion)
        elif usage == 'PrivateTest' or usage == 'Other':
            X_test.append(img_final)
            y_test.append(emotion)
            
    return (np.array(X_train), np.array(y_train)), \
           (np.array(X_val), np.array(y_val)), \
           (np.array(X_test), np.array(y_test))