import os
import cv2
import random
import numpy as np

def extract_random_frames(video_path, output_folder, num_frames=50, min_interval=1):
    """
    Trích xuất các frame ngẫu nhiên từ video với khoảng cách tối thiểu giữa các frame

    Parameters:
        video_path (str): Đường dẫn đến file video
        output_folder (str): Thư mục đầu ra để lưu các frame
        num_frames (int): Số frame cần trích xuất
        min_interval (float): Khoảng cách tối thiểu giữa các frame (giây)
    """
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return

    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Tính toán số frame tối thiểu cần có giữa các frame được chọn
    min_frame_interval = int(fps * min_interval)

    # Kiểm tra nếu video quá ngắn
    if duration < num_frames * min_interval:
        print(f"Video {video_path} quá ngắn để trích xuất {num_frames} frame cách nhau {min_interval} giây")
        return

    # Tạo danh sách các frame có thể chọn
    available_frames = list(range(0, total_frames - min_frame_interval))

    # Chọn các frame ngẫu nhiên
    selected_frames = []
    for _ in range(num_frames):
        if not available_frames:
            break

        # Chọn frame ngẫu nhiên
        idx = random.choice(available_frames)
        selected_frames.append(idx)

        # Loại bỏ các frame xung quanh để đảm bảo khoảng cách
        start = max(0, idx - min_frame_interval)
        end = min(total_frames, idx + min_frame_interval)
        available_frames = [f for f in available_frames if f < start or f > end]

    # Lấy tên video (không bao gồm đuôi mở rộng)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Đọc và lưu các frame đã chọn
    for i, frame_num in enumerate(selected_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            output_filename = f"{video_name}_{i+1:04d}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, frame)

    cap.release()

def process_videos_in_folder(input_folder, output_folder, num_frames=50):
    """
    Xử lý tất cả video trong thư mục đầu vào

    Parameters:
        input_folder (str): Thư mục chứa các video
        output_folder (str): Thư mục đầu ra để lưu các frame
        num_frames (int): Số frame cần trích xuất từ mỗi video
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)

    # Lấy danh sách tất cả file video trong thư mục
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
    video_files = [f for f in os.listdir(input_folder)
                  if f.lower().endswith(video_extensions)]

    # Xử lý từng video
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        print(f"Đang xử lý: {video_file}")
        extract_random_frames(video_path, output_folder, num_frames)

if __name__ == "__main__":
    # Thay đổi các đường dẫn này theo nhu cầu của bạn
    INPUT_FOLDER = "video_tests"  # Thư mục chứa các video
    OUTPUT_FOLDER = "frame_tests"  # Thư mục lưu các frame

    process_videos_in_folder(INPUT_FOLDER, OUTPUT_FOLDER)