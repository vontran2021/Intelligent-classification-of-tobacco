import cv2
import numpy as np
import matplotlib.pyplot as plt
def rgbCurve(img):
    r, g, b = cv2.split(img)
    # 对通道值进行排序
    r_sorted = np.sort(r.flatten())
    g_sorted = np.sort(g.flatten())
    b_sorted = np.sort(b.flatten())
    # 绘制RGB三通道值曲线
    plt.plot(r_sorted, color='red', label='Red')
    plt.plot(g_sorted, color='green', label='Green')
    plt.plot(b_sorted, color='blue', label='Blue')
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    plt.legend()
    # 将绘制的曲线保存为临时图片
    temp_img_path = './ImageProcessPy/img/RGB_plot.png'
    plt.savefig(temp_img_path)
    plt.close()
    return temp_img_path
