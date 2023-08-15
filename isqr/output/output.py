# Date: 2023.8.15
# Author: Steven
"""
處理產生QR code的過程中
所有相關圖片的輸出。
"""

import os
import cv2


class Output:

    def __init__(self, folder_path, level, img_path):
        self.folder_path = folder_path
        self.level = level
        self.img_path = img_path
        
        if not os.path.isdir("isqr/output/{}/{}".format(self.folder_path, self.level)):
            os.mkdir("isqr/output/{}/{}".format(self.folder_path, self.level))
            
    def save_baseline(self, img):
        cv2.imwrite(
            "isqr/output/{}/{}/baseline.png".format(self.folder_path, self.level), img)

    def save_pixel_based_binary(self, img):
        if not os.path.isdir("isqr/output/{}/{}/pixel-based-binary".format(self.folder_path, self.level)):
            os.mkdir(
                "isqr/output/{}/{}/pixel-based-binary".format(self.folder_path, self.level))
        cv2.imwrite("isqr/output/{}/{}/pixel-based-binary/{}".format(self.folder_path,
                    self.level, self.img_path[12:]), img)

    def save_binary(self, img):
        cv2.imwrite("isqr/output/{}/{}/{}".format(self.folder_path,
                    self.level, "binary.png"), img)

    def save_Ideal(self, img):
        cv2.imwrite("isqr/output/{}/{}/{}".format(self.folder_path,
                    self.level, "Ideal_qr.png"), img)

    def save_jordan(self, img):
        if not os.path.isdir("isqr/output/{}/{}/gaussian-jordan".format(self.folder_path, self.level)):
            os.mkdir(
                "isqr/output/{}/{}/gaussian-jordan".format(self.folder_path, self.level))
        cv2.imwrite("isqr/output/{}/{}/gaussian-jordan/{}".format(
            self.folder_path, self.level, self.img_path[12:]), img)

    def save_codeword_type(self, img):
        if not os.path.isdir("isqr/output/{}/{}/type_adjusted".format(self.folder_path, self.level)):
            os.mkdir(
                "isqr/output/{}/{}/type_adjusted".format(self.folder_path, self.level))
        cv2.imwrite("isqr/output/{}/{}/type_adjusted/{}".format(self.folder_path,
                    self.level, self.img_path[12:]), img)

    def save_blending(self, img):
        if not os.path.isdir("isqr/output/{}/{}/blending".format(self.folder_path, self.level)):
            os.mkdir(
                "isqr/output/{}/{}/blending".format(self.folder_path, self.level))
        if not os.path.isdir("isqr/output/{}/{}/blending_H".format(self.folder_path, self.level)):
            os.mkdir(
                "isqr/output/{}/{}/blending_H".format(self.folder_path, self.level))
        cv2.imwrite("isqr/output/{}/{}/blending/{}".format(self.folder_path,
                    self.level, self.img_path[12:]), img)

    def save_blending_H(self, img, subsize, quality_factor):
        if not os.path.isdir("isqr/output/{}/{}/blending_H/{}_{}".format(self.folder_path, self.level, subsize, quality_factor)):
            os.mkdir("isqr/output/{}/{}/blending_H/{}_{}".format(self.folder_path,
                     self.level, subsize, quality_factor))
        cv2.imwrite("isqr/output/{}/{}/blending_H/{}_{}/{}".format(self.folder_path,
                    self.level, subsize, quality_factor, self.img_path[12:]), img)
