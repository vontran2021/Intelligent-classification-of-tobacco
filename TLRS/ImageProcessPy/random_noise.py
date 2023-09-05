import os
import cv2
import random
import numpy as np

# 噪声类型
NOISE_TYPES = ['gaussian', 'salt', 'poisson', 'speckle']
# 加噪声概率
NOISE_PROBS = [0.2, 0.1, 0.1, 0.1]

# 加高斯噪声
def add_gaussian_noise(image):
    mean = 0
    var = random.randint(10, 50)
    std = var ** 0.5
    noisy_image = image.copy()
    h, w, c = image.shape
    np.random.seed(42)
    noise = np.random.normal(mean, std, size=(h, w, c))
    noisy_image = noisy_image.astype(np.float) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


# 加椒盐噪声
def add_salt_noise(image):
    noisy_image = image.copy()
    h, w, c = image.shape
    np.random.seed(42)
    salt = np.random.randint(0, 2, size=(h, w, c)).astype(np.uint8) * 255
    pepper = np.random.randint(0, 2, size=(h, w, c)).astype(np.uint8) * 255
    noisy_image[salt == 255] = 255
    noisy_image[pepper == 255] = 0
    return noisy_image


# 加泊松噪声
def add_poisson_noise(image):
    noisy_image = image.copy()
    h, w, c = image.shape
    np.random.seed(42)
    noise = np.random.poisson(noisy_image, size=(h, w, c))
    noisy_image = np.clip(noise, 0, 255).astype(np.uint8)
    return noisy_image


# 加乘性噪声
def add_speckle_noise(image):
    noisy_image = image.copy()
    h, w, c = image.shape
    np.random.seed(42)
    noise = np.random.randn(h, w, c) * noisy_image
    noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
    return noisy_image


# 对图像随机加噪声
def add_noise(image):
    # 随机选择噪声类型
    noise_type = random.choices(NOISE_TYPES, weights=NOISE_PROBS)[0]
    # 加噪声
    if noise_type == 'gaussian':
        noisy_image = add_gaussian_noise(image)
    elif noise_type == 'salt':
        noisy_image = add_salt_noise(image)
    elif noise_type == 'poisson':
        noisy_image = add_poisson_noise(image)
    elif noise_type == 'speckle':
        noisy_image = add_speckle_noise(image)
    else:
        noisy_image = image
    return noisy_image, noise_type


# 加载图片并随机加噪声
def load_image_with_noise(file_path):
    image = cv2.imread(file_path)
    noisy_image, noise_type = add_noise(image)
    return noisy_image, noise_type


# 批量加噪声并保存
def add_noise_to_images(data_dir):

    file_path = os.path.join(data_dir)
    noisy_image, noise = load_image_with_noise(file_path)
    cv2.imwrite('./ImageProcessPy/Img/noisy_image.jpg', noisy_image)
    return noise
    # print(f"Completed! {i + 1} images have been noise and saved to {output_dir}")
# # 测试
# data_dir = 'D:\data_set\Flue-cured baking stage\DataSet\W\pre_train\Stage10'
# output_dir = 'D:\data_set\Flue-cured baking stage\DataSet\W\jiaqiang\Stage10'
# add_noise_to_images(data_dir, output_dir)
