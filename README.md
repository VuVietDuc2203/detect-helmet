# Project README

## I. Training

1. **Extract Frame từ Video**
   - Sử dụng file `extract_frame.py` để trích xuất frame từ video. Bạn cần thay đổi đường dẫn video trong file cho phù hợp với dự án của mình.
   - Mỗi video sẽ được trích xuất ngẫu nhiên 50 frame.



2. **Gắn nhãn helmet và non_helmet**
   - Dùng công cụ labelling tool để đánh nhãn phần đầu của người trong các frame đã trích xuất.

3. **Cắt Hình Ảnh**
   - Sau khi dự đoán xong, chạy file `crop_img.py` từ thư mục chứa các frame đã dự đoán để cắt các ảnh có ID 2 (helmet) và ID 3 (non_helmet).

4. **Chia Dữ Liệu**
   - Sử dụng file `split_data.py` để chia dữ liệu thành 3 tập: train, test và validation theo tỷ lệ 8:1:1.


5. **Chạy YOLO Classifier**
   - Chạy script `yolo_class.py` để huấn luyện mô hình YOLO.


## II. Validation

1. **Đánh Giá Model**
   - Chạy file `val.py` để đánh giá hiệu quả của mô hình đã huấn luyện.


2. **Dự Đoán trên Ảnh**
   - Chạy `predict.py` để dự đoán tất cả các ảnh trong thư mục đã chỉ định. Kết quả dự đoán sẽ được ghi lên ảnh (ví dụ: `helmet_test_results`).
