import sys
import cv2
import numpy as np

# image_path = "./1.jpg"

def dark_channel(image, patch_size):
    # 计算暗通道图像
    dark_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(dark_channel, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel, top_percentage):
    # 根据暗通道图像估计大气光
    num_pixels = int(dark_channel.size * top_percentage)
    flat_dark_channel = dark_channel.reshape(-1)
    indices = np.argpartition(flat_dark_channel, -num_pixels)[-num_pixels:]
    atmospheric_light = np.max(image.reshape(-1, 3)[indices], axis=0)
    return atmospheric_light

def estimate_transmission(image, atmospheric_light, omega, patch_size):
    # 估计透射率
    normalized_image = image.astype(np.float32) / atmospheric_light.astype(np.float32)
    transmission = 1 - omega * dark_channel(normalized_image, patch_size)
    return transmission

def refine_transmission(image, transmission, epsilon, max_iter):
    # 优化透射率
    refined_transmission = cv2.ximgproc.guidedFilter(image.astype(np.float32), transmission.astype(np.float32), 20, epsilon, max_iter)
    return refined_transmission

def dehaze_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 参数设置
    patch_size = 15
    top_percentage = 0.001
    omega = 0.95
    epsilon = 0.001
    max_iter = 100

    # 去雾处理
    dark_channel_img = dark_channel(image, patch_size)
    atmospheric_light = estimate_atmospheric_light(image, dark_channel_img, top_percentage)
    transmission = estimate_transmission(image, atmospheric_light, omega, patch_size)
    refined_transmission = refine_transmission(image, transmission, epsilon, max_iter)
    dehazed_image = np.zeros_like(image, dtype=np.uint8)
    for c in range(3):
        dehazed_image[:,:,c] = (image[:,:,c].astype(np.float32) - atmospheric_light[c]) / refined_transmission + atmospheric_light[c]
    dehazed_image = np.clip(dehazed_image, 0, 255).astype(np.uint8)

    # return dehazed_image

#     # 保存去雾后的图像
    output_path = "./ImageProcessPy/Img/dehazed_image.jpg"
    cv2.imwrite(output_path, dehazed_image)
    # print(f"去雾后的图像已保存到 {output_path}")
#
# # 调用函数对图像进行去雾处理
# dehaze_image(image_path)
