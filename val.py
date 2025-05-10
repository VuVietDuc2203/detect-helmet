from ultralytics import YOLO
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load model YOLOv8n-cls
model = YOLO("runs/classify/train/weights/best.pt")  # Hoặc thay bằng đường dẫn model của bạn

# Đường dẫn đến thư mục test (cấu trúc: test/helmet/, test/non_helmet/)
test_folder = "data/test"  # Thay bằng đường dẫn thư mục test của bạn

# Danh sách các lớp (phải trùng với thứ tự khi train)
class_names = ["helmet", "non_helmet"]

# Khởi tạo list để lưu ground truth và predictions
true_labels = []
pred_labels = []

# Duyệt qua từng lớp (helmet và non_helmet)
for class_idx, class_name in enumerate(class_names):
    class_folder = os.path.join(test_folder, class_name)

    # Lấy danh sách ảnh trong thư mục lớp hiện tại
    image_files = [f for f in os.listdir(class_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Dự đoán từng ảnh
    for img_file in image_files:
        img_path = os.path.join(class_folder, img_file)

        # Dự đoán
        results = model.predict(img_path)

        # Lấy class dự đoán (class có xác suất cao nhất)
        pred_class_idx = results[0].probs.top1
        pred_class = class_names[pred_class_idx]

        # Lưu ground truth và prediction
        true_labels.append(class_idx)
        pred_labels.append(pred_class_idx)

# Chuyển thành numpy array để tính toán
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

# Tính confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# In classification report (Precision, Recall, F1, Accuracy)
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names))
print(f"Overall Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")  # Lưu confusion matrix
plt.show()