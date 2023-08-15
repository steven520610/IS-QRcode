# Date: 2023.4.7
# Author: Steven
# Usage:

import matplotlib.pyplot as plt
import numpy as np
import cv2


class Qrimg:

    def __init__(self, img_path, size):
        self.img_path = img_path
        self.read_img()
        self.BGR2LAB()
        self.read_mask()

        self.img = Qrimg.resize(self.img, size)
        self.img_luminance = Qrimg.resize(self.img_luminance, size)

    def read_img(self):
        self.img = cv2.imread(self.img_path, flags=cv2.IMREAD_COLOR)

    def BGR2LAB(self):
        self.img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        self.img_luminance = self.img_lab[:, :, 0]

    def read_mask(self):
        self.blendmask = cv2.imread(
            "{}/BlendMask/{}".format(self.img_path[:4], self.img_path[12:]), flags=cv2.IMREAD_GRAYSCALE)

    @classmethod
    def show(cls, img, flag, window_name):
        """_summary_

        Args:
            img (_type_): _description_
            flag (_type_): _description_
        """
        plt.figure(num=window_name)
        # binary
        if flag == "B":
            plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        elif flag == "C":
            # 因為原圖是用cv2.imread讀取進來
            # cv2圖片預設的色彩空間格式為BGR
            # 所以此處轉回RGB才符合imshow的讀法
            img = img[:, :, [2, 1, 0]]
            plt.imshow(img)
        plt.show()
    # 這邊和後面用resize去處理圖片不一樣
    # resize是利用插值去處理
    # 這邊是透過一個一個pixel去assign來達成放大的效果

    @classmethod
    def enlarge(cls, img, multiple):
        """_summary_

        Args:
            img (_type_): _description_
            module_size (_type_): _description_
        """
        size = img.shape[0]
        enlarge_img = np.zeros(
            (size * multiple, size * multiple), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                enlarge_img[i * multiple:(i+1) * multiple,
                            j * multiple: (j+1) * multiple] = img[i][j]
        return enlarge_img

    @classmethod
    def resize(cls, img, size):
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    # 秀出QR code的codeword的index
    @classmethod
    def show_codeword_index(cls, codeword_index, window_name):
        plt.figure(num=window_name)
        plt.imshow(codeword_index, cmap="gray", vmin=0, vmax=255)
        plt.show()
    # 秀出QR code的codeword的type的function

    @classmethod
    def show_codeword_type(cls, codeword_type, window_name):
        plt.figure(num=window_name)
        plt.imshow(codeword_type)
        plt.show()
    # 秀出QR code的codeword的block index的function

    @classmethod
    def show_block_no(cls, block_no, window_name):
        plt.figure(num=window_name)
        plt.imshow(block_no)
        plt.show()
