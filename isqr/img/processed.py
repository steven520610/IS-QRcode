# Date: 2023.4.7
# Author: Steven
# Usage:

import numpy as np
import cv2
import sys
import math

from ..qr import *
from .img import Qrimg


class Process:

    @classmethod
    def OTSU(cls, img_luminance):
        histogram = np.zeros(256)
        for i in range(img_luminance.shape[0]):
            for j in range(img_luminance.shape[1]):
                histogram[img_luminance[i][j]] += 1
        distribution = histogram / np.sum(histogram)

        best_threshold = 0
        minimum_variance = sys.maxsize

        for threshold in range(256):
            # calculate mean
            # --------------
            # lower than threshold
            sum_low = 0
            for i in range(threshold):
                sum_low += distribution[i] * i
            mean_low = sum_low / np.sum(distribution[:threshold])

            # higher than threshold
            sum_high = 0
            for i in range(threshold, 256):
                sum_high += distribution[i] * i
            mean_high = sum_high / np.sum(distribution[threshold:])

            # calculate variance
            # ------------------
            # lower
            variance_low = 0
            for i in range(threshold):
                variance_low += (mean_low - i)**2 * distribution[i]

            # higher
            variance_high = 0
            for i in range(threshold, 256):
                variance_high += (mean_high - i)**2 * distribution[i]

            # sum up
            group_in_variance = variance_low + variance_high

            if group_in_variance < minimum_variance:
                best_threshold = threshold
                minimum_variance = group_in_variance

        _, binarized_img = cv2.threshold(
            img_luminance, best_threshold, 255, cv2.THRESH_BINARY)
        return binarized_img, best_threshold, minimum_variance

    @classmethod
    def module_based_binarization(cls, img, module_size):
        img_size = img.shape[0]
        subimage_size = module_size
        binary_img = np.zeros(
            (img.shape[0] // module_size, img.shape[1] // module_size), dtype=np.uint8)

        # 自己建立Gaussian kernel
        gaussian_kernel = np.zeros((subimage_size, subimage_size))
        center = (subimage_size - 1) // 2
        sigma = 1

        # 設定Gaussian kernel內各個i, j的值
        for i in range(subimage_size):
            for j in range(subimage_size):
                gaussian_kernel[i][j] = 1/(2*math.pi*sigma**2) * \
                    np.exp(-((i-center)**2 + (j-center)**2) / 2*sigma**2)

        # 讓此kernel的總和為1
        gaussian_kernel /= np.sum(gaussian_kernel)

        # 計算各個subimage的binarization的結果後
        # assign給整個subimage，即Module
        for i in range(img_size // subimage_size):
            for j in range(img_size // subimage_size):
                region = img[i*subimage_size:(i+1)*subimage_size,
                             j*subimage_size:(j+1)*subimage_size]
                value = np.round(np.sum(region * gaussian_kernel) / 255)
                binary_img[i][j] = value * 255
        return binary_img

    @classmethod
    def module_based_blending(cls, qr, img, m):
        Ideal = np.zeros(qr.shape, dtype=np.uint8)
        for i in range(qr.shape[0]):
            for j in range(qr.shape[1]):
                # 如果m[i][j]為255(即forbidden area & data codewords的部分)
                # Ideal QR[i][j]即和Standard QR[i][j]相同
                if m[i][j]:
                    Ideal[i][j] = qr[i][j]
                # 若m[i][j]為0(非forbidden area | data codewords)
                # Ideal QR[i][j]即和binarization image相同
                else:
                    Ideal[i][j] = img[i][j]
        return Ideal

    @classmethod
    def GJE(cls, matrix, cols_to_eliminate):
        # Convert the matrix to simplest row-echelon form
        lead = 0
        rowCount, _ = matrix.shape

        for r in range(rowCount):
            if lead >= len(cols_to_eliminate):
                break

            i = r
            while i < rowCount and matrix[i, cols_to_eliminate[lead]] == 0:
                i += 1

            if i < rowCount:
                matrix[[r, i]] = matrix[[i, r]]

            for k in range(rowCount):
                if k != r and matrix[k, cols_to_eliminate[lead]] != 0:
                    matrix[k] = np.logical_xor(matrix[k], matrix[r])

            lead += 1
        # Print the original matrix and its simplest row-echelon form
        return matrix

    @classmethod
    def select_module(cls, canvas: Canvas, args: QrArgs, data: QrData, blendmask,
                      base_qrcode, binary_image):
        extra_bit_stream_lst = []
        RS_bit_stream_lst = []
        count = 0
        g2_count = 0
        for i in range(args.nbg1 + args.nbg2):
            extra_bit_stream_lst.append([])
            RS_bit_stream_lst.append([])
            if i < args.nbg1:
                for j in range(args.nbg1b):
                    if count < data.data_bitstream_length:
                        extra_bit_stream_lst[i].append(
                            "0" * args.nbg1b)
                        RS_bit_stream_lst[i].append(QrData.extra_RS_encoding(
                            extra_bit_stream_lst[i][j], n=args.ndcg1b + args.neceb, k=args.ndcg1b))
                        count += 1
                    else:
                        extra_bit_stream_lst[i].append("0" * (count % args.nbg1b) + "1" + "0" * (
                            args.nbg1b - (count % args.nbg1b) - 1))
                        RS_bit_stream_lst[i].append(QrData.extra_RS_encoding(
                            extra_bit_stream_lst[i][j], n=args.ndcg1b + args.neceb, k=args.ndcg1b))
                        count += 1
            else:
                for j in range(args.nbg2b):
                    if count < data.data_bitstream_length:
                        extra_bit_stream_lst[i].append(
                            "0" * args.nbg2b)
                        RS_bit_stream_lst[i].append(QrData.extra_RS_encoding(
                            extra_bit_stream_lst[i][j], n=args.ndcg2b + args.neceb, k=args.ndcg2b))
                        count += 1
                        g2_count += 1
                    else:
                        extra_bit_stream_lst[i].append("0" * (g2_count % args.nbg2b) + "1" + "0" * (
                            args.nbg2b - (g2_count % args.nbg2b) - 1))
                        RS_bit_stream_lst[i].append(QrData.extra_RS_encoding(
                            extra_bit_stream_lst[i][j], n=args.ndcg2b + args.neceb, k=args.ndcg2b))
                        count += 1
                        g2_count += 1

        module_type_array = np.zeros(
            (canvas.args.size, canvas.args.size), dtype=np.uint8)
        for row in range(canvas.args.size):
            for col in range(canvas.args.size):
                # 換用第三個channel去比較，因為data bit的第一個channel也是0
                # forbidden
                if canvas.codeword_type[row, col, 2] == 0:
                    module_type_array[row, col] = 0
                # data bit
                elif canvas.codeword_type[row, col, 2] == 234:
                    module_type_array[row, col] = 1
                # elif base_qr.codeword_type[row, col, 2] == 13:
                #     module_type_array[row, col] = 2
                else:
                    module_type_array[row, col] = 3

        print("Length of encoded message:{}".format(data.data_bitstream_length))
        print("----------------------------")

        # 先處理位於BlendMask輸出的結果內的Module，因為他們包含最重要的資訊
        # 所以優先處理

        # mask的module取完之後
        # 接著從中間開始，往四周去取module，因為理論上越靠中間，越有可能是ROI的區域
        # 利用每個module和center的距離，去決定哪個module要先取
        distance_array = np.zeros((canvas.args.size, canvas.args.size))
        center = canvas.args.size // 2
        for i in range(canvas.args.size):
            for j in range(canvas.args.size):
                distance_array[i, j] = math.sqrt(
                    (i - center) ** 2 + (j - center) ** 2)
        distance_flatten = distance_array.flatten()
        distance_flatten.sort()
        distance_unique = np.unique(distance_flatten)

        current_length = data.data_bitstream_length

        # 在mask內且被選過的module會被標記為1
        selected_module_in_mask = np.zeros(
            (canvas.args.size, canvas.args.size), dtype=np.uint8)
    # ----------------------------------------------------------------------------------------------- #
        for i in range(args.nbg1 + args.nbg2):
            if i < args.nbg1:
                if current_length >= args.nbg1b:
                    current_length -= args.nbg1b
                    print("There is no control module in block {}.".format(i+1))
                    print("----------------------------")
                    continue
            else:
                if current_length >= args.nbg2b:
                    current_length -= args.nbg2b
                    print("There is no control module in block {}.".format(i+1))
                    print("----------------------------")
                    continue
            # 用來存哪個module會被選來組成basis vector
            control_module_index = []

            # 存basis vector的value
            basis_vector_np = []
            # 觀察用，因為array沒辦法完整顯示
            basis_vector_str = []

            count = 0
            row, col = 0, 0

            # 決定當前的block內有多少module位於mask的位置
            # 若太少module滿足，後面會在從中間往外尋找該block內其他非mask內的點
            # 加到control module內
            valid_mask_count_in_current_block = 0
            for bi in range(canvas.args.size):
                for bj in range(canvas.args.size):
                    if canvas.block_no[bi, bj] == i+1 and blendmask[bi, bj] != 255 and module_type_array[bi, bj] != 1:
                        valid_mask_count_in_current_block += 1

            if i < args.nbg1:
                print("number of control module available in block {}: {} - {} = {}".format(
                    i+1, args.nbg1b, current_length, args.nbg1b - current_length))
                print("valid mask count in block {}: {}".format(
                    i+1, valid_mask_count_in_current_block))
            else:
                print("number of control module available in block {}: {} - {} = {}".format(
                    i+1, args.nbg2b, current_length, args.nbg2b - current_length))
                print("valid mask count in block {}: {}".format(
                    i+1, valid_mask_count_in_current_block))
            print("----------------------------")
            should_break = False
            if i < args.nbg1:
                while count < args.nbg1b - current_length and count != valid_mask_count_in_current_block:
                    for index in range(len(distance_unique)):
                        # print(count)
                        (selected_row, selected_col) = np.where(
                            distance_array == distance_unique[index])
                        n_of_element = len(selected_row)
                        for j in range(n_of_element):
                            # print(count)
                            # 選到function pattern, data bit是本來就不行的關係
                            # 選到padding bit是因為前面BlendMask的結果已經選過了，不能重複選
                            # 因此只能選parity bit
                            if module_type_array[selected_row[j], selected_col[j]] != 3 or \
                                    canvas.block_no[selected_row[j], selected_col[j]] != i+1:
                                continue
                            if blendmask[selected_row[j], selected_col[j]] != 255:
                                control_module_index.append(
                                    int(canvas.module_block_index[i, selected_row[j], selected_col[j]]))
                                module_type_array[selected_row[j],
                                                  selected_col[j]] = 2
                                selected_module_in_mask[selected_row[j],
                                                        selected_col[j]] = 1
                                count += 1

                            if count >= args.nbg1b - current_length:
                                should_break = True
                                break

                        if should_break:
                            break

                # 如果上面loop跑完，mask內的control module不夠時
                # 從中間往外再找非mask的control module
                should_break = False
                while count < args.nbg1b - current_length:
                    for index in range(len(distance_unique)):
                        (selected_row, selected_col) = np.where(
                            distance_array == distance_unique[index])
                        n_of_element = len(selected_row)
                        for j in range(n_of_element):
                            # 選到function pattern, data bit是本來就不行的關係
                            # 選到padding bit是因為前面BlendMask的結果已經選過了，不能重複選
                            # 因此只能選parity bit
                            if module_type_array[selected_row[j], selected_col[j]] != 3 or \
                                    canvas.block_no[selected_row[j], selected_col[j]] != i+1:
                                continue
                            if not selected_module_in_mask[selected_row[j], selected_col[j]]:
                                control_module_index.append(
                                    int(canvas.module_block_index[i, selected_row[j], selected_col[j]]))
                                selected_module_in_mask[selected_row[j],
                                                        selected_col[j]] = 1
                                module_type_array[selected_row[j],
                                                  selected_col[j]] = 2
                                count += 1
                            if count >= args.nbg1b - current_length:
                                should_break = True
                                break

                        if should_break:
                            break
            else:
                while count < args.nbg2b - current_length and count != valid_mask_count_in_current_block:
                    for index in range(len(distance_unique)):
                        (selected_row, selected_col) = np.where(
                            distance_array == distance_unique[index])
                        n_of_element = len(selected_row)
                        for j in range(n_of_element):
                            # 選到function pattern, data bit是本來就不行的關係
                            # 選到padding bit是因為前面BlendMask的結果已經選過了，不能重複選
                            # 因此只能選parity bit
                            if module_type_array[selected_row[j], selected_col[j]] != 3 or \
                                    canvas.block_no[selected_row[j], selected_col[j]] != i+1:
                                continue
                            if blendmask[selected_row[j], selected_col[j]] != 255:
                                control_module_index.append(
                                    int(canvas.module_block_index[i, selected_row[j], selected_col[j]]))
                                selected_module_in_mask[selected_row[j],
                                                        selected_col[j]] = 1
                                module_type_array[selected_row[j],
                                                  selected_col[j]] = 2
                                count += 1

                            if count >= args.nbg2b - current_length:
                                should_break = True
                                break

                        if should_break:
                            break

                # 如果上面loop跑完，mask內的control module不夠時
                # 從中間往外再找非mask的control module
                should_break = False
                while count < args.nbg2b - current_length:
                    for index in range(len(distance_unique)):
                        (selected_row, selected_col) = np.where(
                            distance_array == distance_unique[index])
                        n_of_element = len(selected_row)
                        for j in range(n_of_element):
                            # 選到function pattern, data bit是本來就不行的關係
                            # 選到padding bit是因為前面BlendMask的結果已經選過了，不能重複選
                            # 因此只能選parity bit
                            if module_type_array[selected_row[j], selected_col[j]] != 3 or \
                                    canvas.block_no[selected_row[j], selected_col[j]] != i+1:
                                continue
                            if not selected_module_in_mask[selected_row[j], selected_col[j]]:
                                control_module_index.append(
                                    int(canvas.module_block_index[i, selected_row[j], selected_col[j]]))
                                selected_module_in_mask[selected_row[j],
                                                        selected_col[j]] = 1
                                module_type_array[selected_row[j],
                                                  selected_col[j]] = 2
                                count += 1
                            if count >= args.nbg2b - current_length:
                                should_break = True
                                break

                        if should_break:
                            break

            for row in range(canvas.args.size):
                for col in range(canvas.args.size):
                    if module_type_array[row, col] == 2:
                        canvas.codeword_type[row, col, 0] = 27
                        canvas.codeword_type[row, col, 1] = 184
                        canvas.codeword_type[row, col, 2] = 13
                    elif module_type_array[row, col] == 3:
                        canvas.codeword_type[row, col, 0] = 130
                        canvas.codeword_type[row, col, 1] = 5
                        canvas.codeword_type[row, col, 2] = 5

            for ci in range(len(control_module_index)):
                basis_vector_np.append([])
                basis_vector_str.append("")
                # ** current length **
                if i < args.nbg1:
                    for cj in range((args.ndcg1b + args.neceb) * 8):
                        basis_vector_np[ci].append(
                            int(RS_bit_stream_lst[i][ci+current_length][cj]))
                        basis_vector_str[ci] += RS_bit_stream_lst[i][ci +
                                                                     current_length][cj]
                else:
                    for cj in range((args.ndcg2b + args.neceb) * 8):
                        basis_vector_np[ci].append(
                            int(RS_bit_stream_lst[i][ci+current_length][cj]))
                        basis_vector_str[ci] += RS_bit_stream_lst[i][ci +
                                                                     current_length][cj]

            # transform basis_vector to numpy type
            basis_vector_np = np.array(basis_vector_np)
            basis_vector_np = Process.GJE(
                basis_vector_np, control_module_index)

            basis_vector_str_GJE = []
            for bi in range(basis_vector_np.shape[0]):
                basis_vector_str_GJE.append("")
                for bj in range(basis_vector_np.shape[1]):
                    basis_vector_str_GJE[bi] += str(basis_vector_np[bi][bj])
            # return basis_vector_str_GJE, control_module_index
            # --------------------------------------------------------------------------------- #
            for index in control_module_index:
                (row, col) = np.where(canvas.module_block_index[i] == index)
                vector_index = control_module_index.index(index)
                # 用已經mask完的output去和binary image做比較
                # 再從尚未mask的qr code(base)去更改module
                # 最後再根據更改完的去擺放module已經做mask的動作
                if base_qrcode[row, col] != binary_image[row, col]:
                    # assert i <= 10
                    # 在base(尚未做mask)的QR code做reverse的動作
                    canvas.reverse_base(row, col)

                    if i < args.nbg1:
                        extra_block = QrData.extra_RS_encoding(
                            basis_vector_str_GJE[vector_index][:args.nbg1b], args.ndcg1b + args.neceb, args.ndcg1b)
                        XOR_list = QrData.XOR(
                            data.data_codeword_lst[i] + data.error_codeword_lst[i], extra_block)
                        data.data_codeword_lst[i] = XOR_list[:args.ndcg1b]
                        data.error_codeword_lst[i] = XOR_list[args.ndcg1b:]
                    else:
                        extra_block = QrData.extra_RS_encoding(
                            basis_vector_str_GJE[vector_index][:args.nbg2b], args.ndcg2b + args.neceb, args.ndcg2b)
                        XOR_list = QrData.XOR(
                            data.data_codeword_lst[i] + data.error_codeword_lst[i], extra_block)
                        data.data_codeword_lst[i] = XOR_list[:args.ndcg2b]
                        data.error_codeword_lst[i] = XOR_list[args.ndcg2b:]
            current_length = 0

    @classmethod
    def blending_with_image(cls, canvas: Canvas, XOR_qr: np.ndarray, img: np.ndarray, subsize):
        qr_row, qr_col = XOR_qr.shape
        module_size = img.shape[0] // qr_row
        center = module_size // 2
        expand = subsize // 2

        # Color image
        blending_img = img.copy()
        for row in range(qr_row):
            for col in range(qr_col):
                if canvas.forbidden_area[row, col]:
                    # 不能直接assign XOR_qr的值，因為channel不同
                    if XOR_qr[row, col]:
                        blending_img[row * module_size:(row+1) * module_size,
                                     col * module_size:(col+1) * module_size,
                                     :] = 255
                    else:
                        blending_img[row * module_size:(row+1) * module_size,
                                     col * module_size:(col+1) * module_size,
                                     :] = 0
                else:
                    # if base_qr.codeword_type[row, col,0] == 27:
                    #     continue
                    # else:
                    blending_img[center + (row * module_size) - expand:center + (row * module_size) + expand + 1,
                                 center + (col * module_size) - expand:center + (col * module_size) + expand + 1,
                                 :] = XOR_qr[row, col]
        return blending_img

    @classmethod
    def img_preprocessing(cls, source: Qrimg, threshold, quality_factor):
        preprocessed_img = source.img.copy()
        preprocessed_img = cv2.cvtColor(
            preprocessed_img, cv2.COLOR_BGR2LAB)
        h = threshold * quality_factor

        # 小於threshold的upper bound
        upper_bound = np.round(threshold - h / 2)
        # 大於threshold的lower bound
        lower_bound = np.round(threshold + h / 2)

        # 需要先處理高於threshold，也就是偏白色的部分
        # 因為偏白色的部分如果更新luminance的話，可能會有超過255的問題，導致超過的部分會轉變為黑色
        # ex: 原本的luminance為255，經過處理之後變為270，在ndarray的值會變成15
        # 所以這邊先把經過處理過後會超過255的luminance降低，讓處理過後不會出問題
        # 再做後續的processing
        flag = np.where((source.img_luminance * lower_bound /
                        threshold) >= 255, True, False)
        preprocessed_luminance = np.where(
            flag == True, 255, source.img_luminance)
        # luminance = np.where((luminance * lower_bound / threshold) >= 255, (luminance / lower_bound * threshold), luminance)
        # 每個pixel重新assign luminance，黑白分開處理
        preprocessed_luminance = np.where(preprocessed_luminance <= threshold, np.round(
            preprocessed_luminance * upper_bound / threshold), preprocessed_luminance)
        preprocessed_luminance = np.where((preprocessed_luminance > threshold) & (flag == False), np.round(
            preprocessed_luminance * lower_bound / threshold), preprocessed_luminance)

        preprocessed_img[:, :, 0] = preprocessed_luminance
        preprocessed_img = cv2.cvtColor(
            preprocessed_img, cv2.COLOR_LAB2BGR)
        return preprocessed_img, preprocessed_luminance

    # Ib:Baseline QR Code(after Gauss-Jordan) 1D
    # I:Input Image 3D
    # Ip:Pixel-based Binary Image 1D

    @classmethod
    def blending_H(cls, canvas: Canvas, Ib: np.ndarray, I: np.ndarray, Ip,
                   threshold, subsize):
        qr_row, qr_col = Ib.shape
        module_size = I.shape[0] // qr_row
        center = module_size // 2
        expand = subsize // 2
        # Color image
        blending_img = I.copy()

        for row in range(qr_row):
            for col in range(qr_col):
                if canvas.forbidden_area[row, col]:
                    # 不能直接assign XOR_qr的值，因為channel不同
                    if Ib[row, col]:
                        blending_img[row * module_size:(row+1) * module_size,
                                     col * module_size:(col+1) * module_size,
                                     :] = 255
                    else:
                        blending_img[row * module_size:(row+1) * module_size,
                                     col * module_size:(col+1) * module_size,
                                     :] = 0
                else:
                    Ti = np.sum(Ip[row * module_size:(row+1) * module_size,
                                   col * module_size:(col+1) * module_size]) / module_size ** 2
                    # case 1:
                    if Ti < threshold and Ib[row, col] == 0:
                        continue
                    # case 2:
                    elif Ti > threshold and Ib[row, col] == 0:
                        blending_img[center + (row * module_size) - expand:center + (row * module_size) + expand + 1,
                                     center + (col * module_size) - expand:center + (col * module_size) + expand + 1,
                                     :] = Ib[row, col]
                    # case 3:
                    elif Ti < threshold and Ib[row, col] == 255:
                        blending_img[center + (row * module_size) - expand:center + (row * module_size) + expand + 1,
                                     center + (col * module_size) - expand:center + (col * module_size) + expand + 1,
                                     :] = Ib[row, col]
                    # case 4:
                    elif Ti > threshold and Ib[row, col] == 255:
                        continue
        return blending_img
