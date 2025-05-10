import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import gabor

# Đọc ảnh đầu vào
image_path = 'helmet/mamnonhh1001[T2]_2_1.jpg'  # Đổi thành đường dẫn tới ảnh của bạn
image = io.imread(image_path)

# Chuyển ảnh sang ảnh xám (grayscale) để dễ xử lý
gray_image = color.rgb2gray(image)

# Áp dụng bộ lọc Gabor
# Các tham số có thể điều chỉnh: theta, sigma, frequency, bandwidth
# theta: hướng của bộ lọc, frequency: tần số, sigma: độ rộng của Gaussian envelope
theta = 1.57  # Góc của bộ lọc
frequency = 0.8   # Tần số của bộ lọc
sigma = 1          # Độ rộng của Gaussian

# Áp dụng bộ lọc Gabor
filtered_image, _ = gabor(gray_image, frequency=frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)

# Hiển thị kết quả
plt.figure(figsize=(10, 5))

# Hiển thị ảnh gốc
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Ảnh gốc')
plt.axis('off')

# Hiển thị ảnh sau khi áp dụng bộ lọc Gabor
plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Ảnh sau khi áp dụng Gabor Filter')
plt.axis('off')

plt.show()
