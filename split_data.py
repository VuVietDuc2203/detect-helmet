import splitfolders

# Đường dẫn đến thư mục chứa dữ liệu gốc
input_folder = 'dataset'  # Thay đổi nếu cần

# Đường dẫn đến thư mục đầu ra
output_folder = './data'

# Tỷ lệ chia (train:val:test)
ratio = (0.8, 0.1, 0.1)

# Chia dữ liệu
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=ratio, group_prefix=None)