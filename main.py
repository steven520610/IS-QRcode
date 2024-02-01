# Modified at: 2023.4.4
# Author: Steven
# Usage: main function that start this project.

"""Aesthetic QR code generation
此module實作了結合實例分割模型與高斯消去法，來產生美化過後的QR code。
須透過command line傳入產生QR code時所需要的各種參數
"""
from isqr.qr import QrData
from isqr.qr import QrArgs
from isqr.qr import Canvas
from isqr.img import Qrimg, Process
from isqr.output import Output
import argparse
import sys
import matplotlib.pyplot as plt
import time
import numpy as np


def main():
    # =========================
    # 解析cmd參數
    # =========================
    parser = argparse.ArgumentParser(
        description="Generate aesthetic QR code", epilog="Hope you enjoy!"
    )
    parser.add_argument("v", help="Version, must be 1~40", type=int)
    parser.add_argument("l", help="Correction level, must be LMQH")
    parser.add_argument("m", help="Message want to encode in QR code")

    parser.add_argument("img", help="Image path that embedded in QR code")
    parser.add_argument("o", help="folder path that save the image")
    parser.add_argument(
        "--mask",
        help="Mask use in generating QR code, must be 0~7",
        default=0,
        type=int,
    )

    parser.add_argument(
        "-ms",
        "--module_size",
        help="Determine how many pixels in a module",
        default=13,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--subsize",
        help="Determine subsize of single module",
        default=3,
        type=int,
    )

    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    # 取得
    qrargs = QrArgs(args.v, args.l, args.mask)

    # 處理input message的部分
    qrdata = QrData(qrargs, args.m)
    output = Output(args.o, args.l, args.img)

    # 資料編碼
    qrdata.decide_encode_mode()
    qrdata.message2bitstream()
    qrdata.encode()
    qrdata.rearrange_bitstream_with_type()
    qrdata.add_remainder()

    # 處理建立qrcode時的背景，也就是function pattern等等
    qrcanvas = Canvas(qrargs, qrdata)

    # # 先秀出目前的function_pattern(即哪些地方是forbidden的)
    # print("Show function pattern & forbidden area")
    # plt.subplot(1, 2, 1)
    # Qrimg.show(qrcanvas.image, "B")
    # plt.subplot(1, 2, 2)
    # Qrimg.show(qrcanvas.forbidden_area, "B")

    # 最後再把整個QR code傳入設定的mask做XOR的步驟即完成(function pattern不會做)
    # print("Show QR code before masking")
    # Qrimg.show(qrcanvas.image, "B", "QR code before masking")

    # Baseline qrcode after masking
    base_qrcode = qrcanvas.apply_mask()
    # print("Show QR code after masking")
    # Qrimg.show(base_qrcode, "B", "QR code after masking")

    # 將QR code放大，讓之後可以和image做Blending的操作
    enlarge_base_qrcode = Qrimg.enlarge(base_qrcode, args.module_size)
    output.save_baseline(enlarge_base_qrcode)

    # print("Show codewords' index")
    # Qrimg.show_codeword_index(qrcanvas.codeword_index, "Codewords' Index")

    # print("Show codewords' type, data in blue, padding in green and error correction in red")
    # Qrimg.show_codeword_type(qrcanvas.codeword_type, "Codewords' Type")

    # print("Show codewords' block index")
    # Qrimg.show_block_no(qrcanvas.block_no, "Codewords' Block Index")

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # 後續處理(QR code synthesis)的部分
    # 輸入圖片
    source = Qrimg(args.img, qrargs.size * args.module_size)

    # print("Show forbidden area and data codewords region")
    # Qrimg.show(qrcanvas.m, "B", "Forbidden area and Data codewords region")

    # ------------------------------------------------------------------------------------------------------------------------------------------------- #
    # 原圖之luminance & binary image
    # Pixel-baesd Binary Image
    binary_img_source, old_threshold, variance = Process.OTSU(source.img_luminance)
    print("Threshold:{}, variance:{}".format(old_threshold, variance))
    # print("Show luminance & binary image for source image")
    # Qrimg.show(source.img_luminance, "B", "Luminance image for source image")
    # Qrimg.show(binary_img_source, "B", "Binary image for source image")

    output.save_pixel_based_binary(binary_img_source)

    # Module-Based-Blending
    binary_image = Process.module_based_binarization(
        source.img_luminance, args.module_size
    )
    Ideal_qr = Process.module_based_blending(base_qrcode, binary_image, qrcanvas.m)
    enlarge_binary_image = Qrimg.enlarge(binary_image, args.module_size)
    output.save_binary(enlarge_binary_image)
    enlarge_Ideal_qr = Qrimg.enlarge(Ideal_qr, args.module_size)
    output.save_Ideal(enlarge_Ideal_qr)
    # print("Show image after binarization & blending with masked QR code")
    # Qrimg.show(binary_image, "B", "Image after binarization")
    # Qrimg.show(Ideal_qr, "B", "Blending with masked QR code")

    narrow_blendmask = Qrimg.resize(source.blendmask, qrargs.size)
    # print("Show BlendMask's output after resize.")
    # Qrimg.show(narrow_blendmask, "B", "BlendMask's output")
    # ------------------------------------------------------------------------------------------------------------------------------------------------- #
    Process.select_module(
        qrcanvas, qrargs, qrdata, narrow_blendmask, base_qrcode, binary_image
    )
    qrdata.rearrange_bitstream_without_type()
    qrdata.add_remainder()
    qrcanvas.set_bitstream_without_type()

    XOR_qrcode = qrcanvas.apply_mask()
    # print("QR code after adjustment")
    # Qrimg.show(XOR_qrcode, "B", "QR code after adjustment")
    output.save_jordan(Qrimg.enlarge(XOR_qrcode, args.module_size))

    # print("Show codeword type after adjustment")
    # Qrimg.show_codeword_type(qrcanvas.codeword_type,
    #                          "Codeword type after adjustment")

    output.save_codeword_type(qrcanvas.codeword_type[:, :, [2, 1, 0]])
    # ------------------------------------------------------------------------------------------------------------------------------------------------- #
    blending_qr = Process.blending_with_image(
        qrcanvas, XOR_qrcode, source.img, args.subsize
    )
    output.save_blending(blending_qr)
    # print("QR code after blending with image")
    # Qrimg.show(blending_qr, "C", "QR code after blending with image")
    # plt.show()
    # ------------------------------------------------------------------------------------------------------------------------------------------------- #
    for subsize in range(3, 14, 2):
        for quality_factor in range(0, 51, 5):
            quality_factor /= 100
            preprocessed_img, preprocessed_luminance = Process.img_preprocessing(
                source, old_threshold, quality_factor
            )
            preprocessed_binary, new_threshold, _ = Process.OTSU(
                preprocessed_luminance.astype(np.uint8)
            )
            blending_qr = Process.blending_H(
                qrcanvas,
                XOR_qrcode,
                preprocessed_img,
                preprocessed_binary,
                new_threshold,
                subsize,
            )

            output.save_blending_H(blending_qr, subsize, quality_factor)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("Execution time: {}".format(end_time - start_time))
