# Date: 2023.8.15
# Author: Steven

"""
Create the canvas of the QR code.
Data part will be empty after finishing this script.
Only function pattern will be drawn on the canvas. 
"""

# import package
from .args import QrArgs
from .data import QrData
import numpy as np
import itertools
import math
from copy import deepcopy

# position pattern
position_pattern = np.zeros((7, 7))
position_pattern[1, 1:6] = 255
position_pattern[5, 1:6] = 255
position_pattern[1:6, 1] = 255
position_pattern[1:6, 5] = 255

# alignment pattern
alignment_pattern = np.zeros((5, 5))
alignment_pattern[1:4, 1:4] = 255
alignment_pattern[2][2] = 0

# format information
FORMAT_INFO = {"01000": "1111010110", "01001": "1011100001", "01010": "0110111000", "01011": "0010001111", "01100": "1000111101",
               "01101": "1100001010", "01110": "0001010011", "01111": "0101100100", "00000": "0000000000", "00001": "0100110111",
               "00010": "1001101110", "00011": "1101011001", "00100": "0111101011", "00101": "0011011100", "00110": "1110000101",
               "00111": "1010110010", "11000": "0101001101", "11001": "0001111010", "11010": "1100100011", "11011": "1000010100",
               "11100": "0010100110", "11101": "0110010001", "11110": "1011001000", "11111": "1111111111", "10000": "1010011011",
               "10001": "1110101100", "10010": "0011110101", "10011": "0111000010", "10100": "1101110000", "10101": "1001000111",
               "10110": "0100011110", "10111": "0000101001"}


class Canvas:
    def __init__(self, args: QrArgs, data: QrData):

        self.args = args
        self.data = data

        # Create attributes
        # 宣告之後要用來輸出的圖片
        self.image = np.zeros((args.size, args.size), dtype=np.uint8)
        # 先將原始的qrcode設成灰色
        # 之後可以較方便的觀察設定黑白時
        # 有無發生錯誤
        self.image += 128

        # 建立一個觀察哪些位置不可放置Module的陣列
        self.forbidden_area = np.zeros((args.size, args.size), dtype=np.uint8)

        # 建立一個記錄codeword的index的陣列
        # 用途是觀察codeword在QR code內是怎麼排列的
        self.codeword_index = np.zeros((args.size, args.size))

        # 建立一個module的index的陣列
        # 用途是在後面的codeword adjustment中
        # 決定該module是否需要被調整
        self.module_index = np.zeros((args.size, args.size))

        # 建立一個觀察codeword的type(message, padding, parity)的陣列
        self.codeword_type = np.zeros(
            (args.size, args.size, 3), dtype=np.uint8)

        # 建立用來觀察該module屬於哪個block的陣列
        self.block_no = np.zeros((args.size, args.size), dtype=np.uint8)

        # 建立一個用來觀察每個module屬於該block中的第幾個index的3D array
        self.module_block_index = np.zeros(
            (args.nbg1 + args.nbg2, args.size, args.size), dtype=np.uint32)

        # ===============================
        # 以下新增QR code的function pattern
        # ===============================
        self.set_function_pattern()

        # m是要取出function pattern以及data codewords的位置
        # 讓後面與圖片做Blending的時候，區分出哪些地方會用standard QR code
        # 哪些地方會用原圖做Blending
        # 這邊因為設定function pattern時就已經有設定forbidden_area了
        # 所以只要在後面place module時，額外加上data codewords的部分即可
        self.m = deepcopy(self.forbidden_area)

        forbidden_count = sum(sum(self.forbidden_area == 255))
        self.module_count = args.size ** 2 - forbidden_count

        # module count every block
        self.mceb = [0] * (self.args.nbg1 + self.args.nbg2)

        self.set_bitstream_with_type()
        self.set_formation_pattern()
        if self.args.version >= 7:
            self.set_version_pattern()

    # function的部分
    # define用來設定pixel的黑白的function
    # 利用set_black, set_white，而不是每次都用
    # self.image[row][col] = 255
    # self.image[row][col] = 0
    def set_black(self, row, col):
        self.image[row, col] = 0

    def set_white(self, row, col):
        self.image[row, col] = 255

    def reverse_base(self, row, col):
        if self.image[row, col] == 0:
            self.image[row, col] = 255
        else:
            self.image[row, col] = 0

    # ====================================
    # 以下新增QR code的function pattern的細項
    # ===================================
    def set_position_pattern(self):
        # 左上
        self.image[0:7, 0:7] = position_pattern
        self.forbidden_area[0:7, 0:7] = 255
        # 右上
        self.image[0:7, self.args.size-7:] = position_pattern
        self.forbidden_area[0:7, self.args.size-7:] = 255
        # 左下
        self.image[self.args.size-7:, 0:7] = position_pattern
        self.forbidden_area[self.args.size-7:, 0:7] = 255

    # 左上、又上、左下的三個回字的 分隔圖案
    def set_seperator_pattern(self):
        for row in range(8):
            # 左上
            self.set_white(row, 7)
            self.forbidden_area[row, 7] = 255

            # 右上
            self.set_white(row, -8)
            self.forbidden_area[row, -8] = 255

            # 左下
            self.set_white(self.args.size-row-1, 7)
            self.forbidden_area[self.args.size-row-1, 7] = 255

        for col in range(8):
            # 左上
            self.set_white(7, col)
            self.forbidden_area[7, col] = 255

            # 右上
            self.set_white(7, self.args.size-col-1)
            self.forbidden_area[7, self.args.size-col-1] = 255

            # 左下
            self.set_white(-8, col)
            self.forbidden_area[-8, col] = 255
            
    # 一個黑點而已
    def set_dark_module(self):
        self.set_black(-8, 8)
        self.forbidden_area[-8, 8] = 255

    # 黑白相間、一整排的 定時資訊
    def set_timing_pattern(self):
        # 要先設定完alignment_pattern，再處理此處的forbidden_area
        # version 1除外，因為version 1 沒有alignment_pattern
        for col in range(8, self.args.size-8, 2):
            self.set_black(6, col)
            self.set_white(6, col+1)
        for row in range(8, self.args.size-8, 2):
            self.set_black(row, 6)
            self.set_white(row+1, 6)
        # 要注意這邊沒有先assign forbidden area
        # 是因為要處理alignment pattern的緣故。

        if self.args.version == 1:
            self.forbidden_area[6, :] = 255
            self.forbidden_area[:, 6] = 255

    # 設定遍佈在整個QR code內的較小的回字 校正圖塊
    def set_alignment_pattern(self):
        # 版本1無
        if self.args.version == 1:
            return
        possible_center = []
        # 笛卡爾乘積
        for i in itertools.product(self.args.possible_center, self.args.possible_center):
            possible_center.append(i)
        # example output: [(6, 6), (6, 30), (30, 6), (30, 30)]
        # 決定哪些點不可當對齊圖案的中心
        for row, col in possible_center:
            if not self.forbidden_area[row][col]:
                self.image[row-2:row+3, col-2:col+3] = alignment_pattern
                self.forbidden_area[row-2:row+3, col-2:col+3] = 255

        # **
        self.forbidden_area[6, :] = 255
        self.forbidden_area[:, 6] = 255

    # 預留之後要設定版本資訊、格式資訊的地方為forbidden_area的function
    # 並且都先設定為黑色的點，方便辨識
    def reserve_formation_pattern(self):
        # 因為會跳來跳去，所以分開處理
        # 左上
        # 0 ~ 5
        for i in range(6):
            self.forbidden_area[8][i] = 255
            self.set_black(8, i)
        # 6, 7
        for i in range(2):
            self.forbidden_area[8][i+7] = 255
            self.set_black(8, i+7)
        # 8
        self.forbidden_area[7][8] = 255
        self.set_black(7, 8)
        # 9 ~ E
        for i in range(6):
            self.forbidden_area[5-i][8] = 255
            self.set_black(5-i, 8)

        # 左下 & 右上
        # 0 ~ 6
        for i in range(7):
            self.forbidden_area[self.args.size-i-1][8] = 255
            self.set_black(self.args.size-i-1, 8)
        # 7 ~ E
        for i in range(8):
            self.forbidden_area[8][self.args.size-8+i] = 255
            self.set_black(8, self.args.size-8+i)

    def reserve_version_pattern(self):
        if self.args.version < 7:
            return

        for i in range(6):
            for j in range(3):
                self.set_black(self.args.size-9-j, 5-i)
                self.set_black(5-i, self.args.size-9-j)
                self.forbidden_area[self.args.size-9-j][5-i] = 255
                self.forbidden_area[5-i][self.args.size-9-j] = 255

    # ---------------------------------------------------------------------------------- #
    def set_function_pattern(self):
        self.set_position_pattern()
        self.set_seperator_pattern()
        self.set_dark_module()
        self.set_timing_pattern()
        self.set_alignment_pattern()
        # formation, version pattern放置完module後才會設定
        # 但還是要先預留這兩個pattern，以免放置module時放到這些位置上
        # 之後再用set_formation_pattern, set_version_pattern
        self.reserve_formation_pattern()
        self.reserve_version_pattern()
    # ---------------------------------------------------------------------------------- #

    def set_bitstream_with_type(self):
        # 對codeword進行編號並放置Module
        # 一開始設定的起始點：右下角
        base_row, base_col = self.args.size-1, self.args.size-1
        next_row, next_col = base_row, base_col
        downward = False
        index = 0
        for i in range(self.module_count):
            current_row, current_col, next_row, next_col, base_row, base_col, downward, index = self.check_forbidden(
                next_row, next_col, base_row, base_col, self.args.size, downward, index)
            self.place_module(current_row, current_col,
                              self.data.interleaved_bitstream[i])
            # 因為remainder_bit的關係
            if i >= self.args.dpppe:
                continue
            # 這邊是為了後面做module adjustment時設計的
            self.codeword_index[current_row, current_col] = (i // 8)
            self.module_index[current_row, current_col] = i

            # 處理module在該位置屬於哪一個block
            if i < self.args.data_capacity:
                block_num = (i // 8 % (self.args.nbg1 + self.args.nbg2)) + 1
                if i >= self.args.ndcg1b * (self.args.nbg1 + self.args.nbg2) * 8:
                    block_num += self.args.nbg1
                self.block_no[current_row, current_col] = block_num
            else:
                block_num = (
                    i // 8 + self.args.nbg1) % (self.args.nbg1 + self.args.nbg2) + 1
                self.block_no[current_row, current_col] = block_num

            # 設定該module在該位置屬於該block中的第幾個index
            block = int(self.data.bfem[i])
            self.module_block_index[block-1, current_row,
                                    current_col] = self.mceb[block-1]
            self.mceb[block-1] += 1

            # 處理每個Module是屬於data, padding or error correction
            if self.data.type_codeword[i // 8] == 1:
                self.codeword_type[current_row, current_col, 0] = 0
                self.codeword_type[current_row, current_col, 1] = 255
                self.codeword_type[current_row, current_col, 2] = 234
                # data codewords的地方要另外做處理，目的是要讓後面能做Module-Based Blending
                self.m[current_row, current_col] = 255
            elif self.data.type_codeword[i // 8] == 2:
                self.codeword_type[current_row, current_col, 0] = 27
                self.codeword_type[current_row, current_col, 1] = 184
                self.codeword_type[current_row, current_col, 2] = 13
            else:
                self.codeword_type[current_row, current_col, 0] = 130
                self.codeword_type[current_row, current_col, 1] = 5
                self.codeword_type[current_row, current_col, 2] = 5

    def check_forbidden(self, row, col, base_row, base_col, size, downward, index):
        # next
        # 要放置的bit抵達上界時的處理
        if row < 0:
            base_row = 0
            # 檢查最後一組codeword時，若base_col - 2 的話，會重新跑到最右邊那一行，會影響到最一開始的codeword
            # 所以如果base_col為第二行時，把base_col設定到第一行就好
            if base_col == 1:
                base_col = 0
            # 正常的情況是每兩行為一個單位
            else:
                base_col = base_col - 2  # ***
            downward = True
            row = base_row
            col = base_col
            # 每次重新設定基準的row, col，index都要歸零
            index = 0  # ***

        # 要放置的bit抵達下界
        if row == size:
            base_row = size - 1
            # 檢查最後一組codeword時，若base_col - 2 的話，會重新跑到最右邊那一行，會影響到最一開始的codeword
            # 所以如果base_col為第二行時，把base_col設定到第一行就好
            if base_col == 1:
                base_col = 0
            # 正常的情況是每兩行為一個單位
            else:
                base_col = base_col - 2  # ***
            downward = False
            row = base_row
            col = base_col
            index = 0
            
        # 每進到這個function一次都會更新index
        index += 1

        is_forbid = 0
        if self.forbidden_area[row][col]:
            is_forbid = 1
        # 如果要放置的地方是forbidden，就要遞迴地尋找下一個可放的位置
        if is_forbid:
            if base_col != 0:
                if downward:
                    row = base_row + math.floor(index/2)
                else:
                    row = base_row - math.floor(index/2)
                if base_col == 6:
                    base_col -= 1
                col = base_col - (index % 2)
            else:
                if downward:
                    row += 1
                else:
                    row -= 1
            return self.check_forbidden(row, col, base_row, base_col, self.args.size, downward, index)
        # 如果要放置的地方沒有問題，就assign下一個bit要放的地方
        # 回到前面的loop內執行下一個bit
        # next_row, next_col 到這邊才會宣告
        else:
            if base_col != 0:
                if downward:
                    next_row = base_row + math.floor(index/2)
                else:
                    next_row = base_row - math.floor(index/2)
                next_col = base_col - (index % 2)
            else:
                if downward:
                    next_row = row + 1
                else:
                    next_row = row - 1
                next_col = 0

        # 回傳用來對codeword_index賦值的變數
        current_row, current_col = row, col
        return current_row, current_col, next_row, next_col, base_row, base_col, downward, index

    # 將data(bit)放置到QR code中
    def place_module(self, row, col, bit):
        # ***
        # 在QR code中
        # white pixel的bit為0
        # black pixel的bit為1
        # 一定要特別注意！
        # 容易和imshow中的白色(255)、黑色(0)搞混！
        # ***
        if bit == "1":
            self.set_black(row, col)
        else:
            self.set_white(row, col)
    # ---------------------------------------------------------------------------------- #

    def set_formation_pattern(self):
        # correction_bit
        if self.args.level == "L":
            format_bits = "01"
        elif self.args.level == "M":
            format_bits = "00"
        elif self.args.level == "Q":
            format_bits = "11"
        else:
            format_bits = "10"
        # mask_bit
        if self.args.mask == 0:
            format_bits += "000"
        elif self.args.mask == 1:
            format_bits += "001"
        elif self.args.mask == 2:
            format_bits += "010"
        elif self.args.mask == 3:
            format_bits += "011"
        elif self.args.mask == 4:
            format_bits += "100"
        elif self.args.mask == 5:
            format_bits += "101"
        elif self.args.mask == 6:
            format_bits += "110"
        elif self.args.mask == 7:
            format_bits += "111"
        # 處理完後和原本info內就已有的format_info的error correction結合
        format_bits += FORMAT_INFO[format_bits]
        # 固定的，32種都用同一個
        format_mask = "101010000010010"
        final_format_bits = bin(int(format_bits, 2) ^ int(format_mask, 2))[
            2:].rjust(15, "0")

        # 因為會跳來跳去，所以分開處理
        # 左上
        # 0 ~ 5
        for i in range(6):
            # bits的部分，黑色為1、白色為0
            if final_format_bits[i] == "1":
                self.set_black(8, i)
            else:
                self.set_white(8, i)
            self.forbidden_area[8][i] = 255
        # 6, 7
        for i in range(2):
            if final_format_bits[i+6] == "1":
                self.set_black(8, i+7)
            else:
                self.set_white(8, i+7)
            self.forbidden_area[8][i+7] = 255
        # 8
        if final_format_bits[8] == 1:
            self.set_black(7, 8)
        else:
            self.set_white(7, 8)
        self.forbidden_area[7][8] = 255
        # 9 ~ E
        for i in range(6):
            if final_format_bits[i+9] == "1":
                self.set_black(5-i, 8)
            else:
                self.set_white(5-i, 8)
            self.forbidden_area[5-i][8] = 255

        # 左下 & 右上
        # 0 ~ 6
        for i in range(7):
            if final_format_bits[i] == "1":
                self.set_black(self.args.size-i-1, 8)
            else:
                self.set_white(self.args.size-i-1, 8)
            self.forbidden_area[self.args.size-i-1][8] = 255
        # 7 ~ E
        for i in range(8):
            if final_format_bits[i+7] == "1":
                self.set_black(8, self.args.size-8+i)
            else:
                self.set_white(8, self.args.size-8+i)
            self.forbidden_area[8][self.args.size-8+i] = 255

    # 這邊最後才會使用加上去
    # 所以一開始會先使用reserve_version_pattern來保留forbidden_area
    def set_version_pattern(self):
        version_bits = self.args.version_info
        for i in range(6):
            for j in range(3):
                if version_bits[i*3+j] == "1":
                    self.set_black(self.args.size-9-j, 5-i)
                    self.set_black(5-i, self.args.size-9-j)
                    self.forbidden_area[self.args.size-9-j][5-i] = 255
                    self.forbidden_area[5-i][self.col-9-j] = 255
                else:
                    self.set_white(self.args.size-9-j, 5-i)
                    self.set_white(5-i, self.args.size-9-j)
                    self.forbidden_area[self.args.size-9-j][5-i] = 255
                    self.forbidden_area[5-i][self.args.size-9-j] = 255

    # ---------------------------------------------------------------------------------- #
    def apply_mask(self):
        # 每次都先初始化成self.image
        # **
        # 注意這邊要用self.image.copy()
        # 否則的話mask_image會指到和self.image相同的記憶體空間
        # 更改mask_image的話就會更改到self.image
        # (所以當初在檢查的時候才會發現到bit_stream怎麼和理碖上的不太一樣)
        # **

        # ex:
        # a = np.array([1,2,3,4])
        # b = a
        # c = a.copy()
        # b[0] = 10
        # c[0] = 5
        # print(a) >> [10, 2, 3, 4]
        # print(c) >> [5, 2, 3, 4]
        mask_image = self.image.copy()

        def reverse_mask(row, col):
            if mask_image[row][col] == 0:
                mask_image[row][col] = 255
            else:
                mask_image[row][col] = 0

        if self.args.mask == 0:
            for row in range(self.args.size):
                for col in range(self.args.size):
                    if self.forbidden_area[row][col]:
                        continue
                    if (row+col) % 2 == 0:
                        reverse_mask(row, col)
        elif self.args.mask == 1:
            for row in range(self.args.size):
                for col in range(self.args.size):
                    if self.forbidden_area[row][col]:
                        continue
                    if row % 2 == 0:
                        reverse_mask(row, col)
        elif self.args.mask == 2:
            for row in range(self.args.size):
                for col in range(self.args.size):
                    if self.forbidden_area[row][col]:
                        continue
                    if col % 3 == 0:
                        reverse_mask(row, col)
        elif self.args.mask == 3:
            for row in range(self.args.size):
                for col in range(self.args.size):
                    if self.forbidden_area[row][col]:
                        continue
                    if (row + col) % 3 == 0:
                        reverse_mask(row, col)
        elif self.args.mask == 4:
            for row in range(self.args.size):
                for col in range(self.args.size):
                    if self.forbidden_area[row][col]:
                        continue
                    if (col // 3 + row // 2) % 2 == 0:
                        reverse_mask(row, col)
        elif self.args.mask == 5:
            for row in range(self.args.size):
                for col in range(self.args.size):
                    if self.forbidden_area[row][col]:
                        continue
                    if ((row * col) % 2 + (row * col) % 3) == 0:
                        reverse_mask(row, col)
        elif self.args.mask == 6:
            for row in range(self.args.size):
                for col in range(self.args.size):
                    if self.forbidden_area[row][col]:
                        continue
                    if (((row * col) % 2) + ((row * col) % 3)) % 2 == 0:
                        reverse_mask(row, col)
        elif self.args.mask == 7:
            for row in range(self.args.size):
                for col in range(self.args.size):
                    if self.forbidden_area[row][col]:
                        continue
                    if (((row + col) % 2) + ((row * col) % 3)) % 2 == 0:
                        reverse_mask(row, col)
        # 最後回傳的是此function一開始宣告的mask_image
        # 不影響原始的image
        return mask_image

    # 這邊和後面用resize去處理圖片不一樣
    # resize是利用插值去處理
    # 這邊是透過一個一個pixel去assign來達成放大的效果
    def enlarge(self, image, multiple):
        size = image.shape[0]
        enlarge_image = np.zeros(
            (size * multiple, size * multiple), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                enlarge_image[i * multiple:(i+1) * multiple,
                              j * multiple: (j+1) * multiple] = image[i][j]
        return enlarge_image

    def set_bitstream_without_type(self):
        # 對codeword進行編號並放置Module
        # 一開始設定的起始點：右下角
        base_row, base_col = self.args.size-1, self.args.size-1
        next_row, next_col = base_row, base_col
        downward = False
        index = 0
        for i in range(self.module_count):
            current_row, current_col, next_row, next_col, base_row, base_col, downward, index = self.check_forbidden(
                next_row, next_col, base_row, base_col, self.args.size, downward, index)
            self.place_module(current_row, current_col,
                              self.data.interleaved_bitstream[i])
