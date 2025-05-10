
from ultralytics import YOLO
import os
import cv2

# Load mô hình YOLO
model = YOLO("runs/classify/train/weights/best.pt")  # Thay bằng đường dẫn đến model của bạn

# Đường dẫn folder chứa ảnh đầu vào
input_folder = "helmet_test"  # Thay bằng đường dẫn folder ảnh của bạn

# Đường dẫn folder để lưu kết quả
output_folder = "helmet_test_results"  # Thay bằng tên folder bạn muốn lưu kết quả

# Tạo folder output nếu chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lấy danh sách tất cả file ảnh trong folder
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
image_files = [f for f in os.listdir(input_folder)
              if os.path.isfile(os.path.join(input_folder, f))
              and os.path.splitext(f)[1].lower() in image_extensions]

# Duyệt qua từng ảnh và dự đoán
for image_file in image_files:
    # Đường dẫn đầy đủ đến ảnh
    image_path = os.path.join(input_folder, image_file)

    # Dự đoán
    results = model.predict(source=image_path, save=False, save_txt=False, show=False)

    # Lưu ảnh kết quả
    for i, r in enumerate(results):
        # Tạo tên file output
        output_path = os.path.join(output_folder, f"result_{image_file}")

        # Lấy ảnh đã được vẽ bounding boxes và labels
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        # Lưu ảnh
        cv2.imwrite(output_path, im)

    print(f"Đã xử lý và lưu: {output_path}")

print("Hoàn thành xử lý tất cả ảnh!")

