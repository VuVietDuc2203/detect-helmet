import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from skimage.filters import gabor_kernel
import scipy.ndimage as ndi


def apply_gabor_filter(image_path):
    # Đọc ảnh từ đường dẫn
    img = io.imread(image_path)

    # Chuyển ảnh sang grayscale
    gray_img = color.rgb2gray(img)

    # Tạo các Gabor kernels với tham số từ code gốc
    theta = 4
    frequency = (0.1, 0.5, 0.8)
    sigma = (1, 3, 5)
    bandwidth = (0.3, 0.7, 1)

    kernels = []
    kernel_params = []  # Lưu trữ thông số của từng kernel

    for t in range(theta):
        theta_val = t / float(theta) * np.pi
        for f in frequency:
            for s in sigma:
                kernel = gabor_kernel(f, theta=theta_val, sigma_x=s, sigma_y=s)
                kernels.append(kernel)
                kernel_params.append({
                    'theta': theta_val,
                    'frequency': f,
                    'sigma': s,
                    'bandwidth': None
                })
            for b in bandwidth:
                kernel = gabor_kernel(f, theta=theta_val, bandwidth=b)
                kernels.append(kernel)
                kernel_params.append({
                    'theta': theta_val,
                    'frequency': f,
                    'sigma': None,
                    'bandwidth': b
                })

    # Áp dụng từng kernel lên ảnh
    results = []
    for k, (kernel, params) in enumerate(zip(kernels, kernel_params)):
        # Lấy phần thực và ảo của kernel
        real = ndi.convolve(gray_img, np.real(kernel), mode='wrap')
        imag = ndi.convolve(gray_img, np.imag(kernel), mode='wrap')

        # Tính độ lớn
        magnitude = np.sqrt(real ** 2 + imag ** 2)

        results.append({
            'kernel_num': k,
            'real': real,
            'imag': imag,
            'magnitude': magnitude,
            'kernel_params': params
        })

    return gray_img, results


def visualize_gabor_results(gray_img, results, n_cols=4):
    n_rows = int(np.ceil(len(results) / n_cols)) + 1
    plt.figure(figsize=(20, 5 * n_rows))

    # Hiển thị ảnh gốc
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Hiển thị các kết quả Gabor
    for i, result in enumerate(results):
        plt.subplot(n_rows, n_cols, i + 2)
        plt.imshow(result['magnitude'], cmap='gray')
        title = f"Kernel {result['kernel_num']}\n"
        title += f"θ={result['kernel_params']['theta']:.2f} "
        title += f"f={result['kernel_params']['frequency']:.1f}\n"
        if result['kernel_params']['sigma'] is not None:
            title += f"σ={result['kernel_params']['sigma']:.1f} "
        if result['kernel_params']['bandwidth'] is not None:
            title += f"bw={result['kernel_params']['bandwidth']:.1f}"
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Đường dẫn ảnh của bạn - THAY ĐỔI ĐƯỜNG DẪN NÀY
image_path = 'helmet/mamnonhh1001[T2]_2_1.jpg'  # Thay thế bằng đường dẫn ảnh của bạn

# Áp dụng Gabor filter và hiển thị kết quả
gray_img, gabor_results = apply_gabor_filter(image_path)
visualize_gabor_results(gray_img, gabor_results)

# In thông tin thống kê về các đặc trưng Gabor
print("\nGabor Features Summary:")
for i, result in enumerate(gabor_results[:5]):  # Hiển thị 5 kết quả đầu làm ví dụ
    print(f"\nKernel {i}:")
    print(f"Parameters: θ={result['kernel_params']['theta']:.2f}, "
          f"f={result['kernel_params']['frequency']:.1f}, "
          f"σ={result['kernel_params']['sigma'] if result['kernel_params']['sigma'] is not None else 'N/A'}, "
          f"bw={result['kernel_params']['bandwidth'] if result['kernel_params']['bandwidth'] is not None else 'N/A'}")
    print(f"Real part - Mean: {np.mean(result['real']):.4f}, Std: {np.std(result['real']):.4f}")
    print(f"Imag part - Mean: {np.mean(result['imag']):.4f}, Std: {np.std(result['imag']):.4f}")
    print(f"Magnitude - Mean: {np.mean(result['magnitude']):.4f}, Std: {np.std(result['magnitude']):.4f}")