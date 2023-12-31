# Date: 2023.8.15
# Author: Steven

"""
根據ISO 18004所定義的規範，處理QR code的版本、容錯等級下
其相對應的各種資訊。
"""

from .version import QrVersion

# 格式資訊包含：
# 容錯等級：4種 (前兩個bits)
# 遮罩編碼：8種 (後三個bits)
# +
# 錯誤糾正碼


# 各 版本&容錯 可容納的data bits數量(不包含error correction bits)
DATA_CAPACITY = {(1, "L"): 152, (1, "M"): 128, (1, "Q"): 104, (1, "H"): 72,
                 (2, "L"): 272, (2, "M"): 224, (2, "Q"): 176, (2, "H"): 128,
                 (3, "L"): 440, (3, "M"): 352, (3, "Q"): 272, (3, "H"): 208,
                 (4, "L"): 640, (4, "M"): 512, (4, "Q"): 384, (4, "H"): 288,
                 (5, "L"): 864, (5, "M"): 688, (5, "Q"): 496, (5, "H"): 368,
                 (6, "L"): 1088, (6, "M"): 864, (6, "Q"): 608, (6, "H"): 480,
                 (7, "L"): 1248, (7, "M"): 992, (7, "Q"): 704, (7, "H"): 528,
                 (8, "L"): 1552, (8, "M"): 1232, (8, "Q"): 880, (8, "H"): 688,
                 (9, "L"): 1856, (9, "M"): 1456, (9, "Q"): 1056, (9, "H"): 800,
                 (10, "L"): 2192, (10, "M"): 1728, (10, "Q"): 1232, (10, "H"): 976,
                 (11, "L"): 2592, (11, "M"): 2032, (11, "Q"): 1440, (11, "H"): 1120,
                 (12, "L"): 2960, (12, "M"): 2320, (12, "Q"): 1648, (12, "H"): 1264,
                 (13, "L"): 3424, (13, "M"): 2672, (13, "Q"): 1952, (13, "H"): 1440,
                 (14, "L"): 3688, (14, "M"): 2920, (14, "Q"): 2088, (14, "H"): 1576,
                 (15, "L"): 4184, (15, "M"): 3320, (15, "Q"): 2360, (15, "H"): 1784,
                 (16, "L"): 4712, (16, "M"): 3624, (16, "Q"): 2600, (16, "H"): 2024,
                 (17, "L"): 5176, (17, "M"): 4056, (17, "Q"): 2936, (17, "H"): 2264,
                 (18, "L"): 5768, (18, "M"): 4504, (18, "Q"): 3176, (18, "H"): 2504,
                 (19, "L"): 6360, (19, "M"): 5016, (19, "Q"): 3560, (19, "H"): 2728,
                 (20, "L"): 6888, (20, "M"): 5352, (20, "Q"): 3880, (20, "H"): 3080,
                 (21, "L"): 7456, (21, "M"): 5712, (21, "Q"): 4096, (21, "H"): 3248,
                 (22, "L"): 8048, (22, "M"): 6256, (22, "Q"): 4544, (22, "H"): 3536,
                 (23, "L"): 8752, (23, "M"): 6880, (23, "Q"): 4912, (23, "H"): 3712,
                 (24, "L"): 9392, (24, "M"): 7312, (24, "Q"): 5312, (24, "H"): 4112,
                 (25, "L"): 10208, (25, "M"): 8000, (25, "Q"): 5744, (25, "H"): 4304,
                 (26, "L"): 10960, (26, "M"): 8496, (26, "Q"): 6032, (26, "H"): 4768,
                 (27, "L"): 11744, (27, "M"): 9024, (27, "Q"): 6464, (27, "H"): 5024,
                 (28, "L"): 12248, (28, "M"): 9544, (28, "Q"): 6968, (28, "H"): 5288,
                 (29, "L"): 13048, (29, "M"): 10136, (29, "Q"): 7288, (29, "H"): 5608,
                 (30, "L"): 13880, (30, "M"): 10984, (30, "Q"): 7880, (30, "H"): 5960,
                 (31, "L"): 14744, (31, "M"): 11640, (31, "Q"): 8264, (31, "H"): 6344,
                 (32, "L"): 15640, (32, "M"): 12328, (32, "Q"): 8920, (32, "H"): 6760,
                 (33, "L"): 16568, (33, "M"): 13048, (33, "Q"): 9368, (33, "H"): 7208,
                 (34, "L"): 17528, (34, "M"): 13800, (34, "Q"): 9848, (34, "H"): 7688,
                 (35, "L"): 18448, (35, "M"): 14496, (35, "Q"): 10288, (35, "H"): 7888,
                 (36, "L"): 19472, (36, "M"): 15312, (36, "Q"): 10832, (36, "H"): 8432,
                 (37, "L"): 20528, (37, "M"): 15936, (37, "Q"): 11408, (37, "H"): 8768,
                 (38, "L"): 21616, (38, "M"): 16816, (38, "Q"): 12016, (38, "H"): 9136,
                 (39, "L"): 22496, (39, "M"): 17728, (39, "Q"): 12656, (39, "H"): 9776,
                 (40, "L"): 23648, (40, "M"): 18672, (40, "Q"): 13328, (40, "H"): 10208,
                 }

# 各 版本&容錯 對應的每個block所需要的error codewords 的數量
ERROR_CODEWORD = {(1, "L"): 7, (1, "M"): 10, (1, "Q"): 13, (1, "H"): 17,
                  (2, "L"): 10, (2, "M"): 16, (2, "Q"): 22, (2, "H"): 28,
                  (3, "L"): 15, (3, "M"): 26, (3, "Q"): 18, (3, "H"): 22,
                  (4, "L"): 20, (4, "M"): 18, (4, "Q"): 26, (4, "H"): 16,
                  (5, "L"): 26, (5, "M"): 24, (5, "Q"): 18, (5, "H"): 22,
                  (6, "L"): 18, (6, "M"): 16, (6, "Q"): 24, (6, "H"): 28,
                  (7, "L"): 20, (7, "M"): 18, (7, "Q"): 18, (7, "H"): 26,
                  (8, "L"): 24, (8, "M"): 22, (8, "Q"): 22, (8, "H"): 26,
                  (9, "L"): 30, (9, "M"): 22, (9, "Q"): 20, (9, "H"): 24,
                  (10, "L"): 18, (10, "M"): 26, (10, "Q"): 24, (10, "H"): 28,
                  (11, "L"): 20, (11, "M"): 30, (11, "Q"): 28, (11, "H"): 24,
                  (12, "L"): 24, (12, "M"): 22, (12, "Q"): 26, (12, "H"): 28,
                  (13, "L"): 26, (13, "M"): 22, (13, "Q"): 24, (13, "H"): 22,
                  (14, "L"): 30, (14, "M"): 24, (14, "Q"): 20, (14, "H"): 24,
                  (15, "L"): 22, (15, "M"): 24, (15, "Q"): 30, (15, "H"): 24,
                  (16, "L"): 24, (16, "M"): 28, (16, "Q"): 24, (16, "H"): 30,
                  (17, "L"): 28, (17, "M"): 28, (17, "Q"): 28, (17, "H"): 28,
                  (18, "L"): 30, (18, "M"): 26, (18, "Q"): 28, (18, "H"): 28,
                  (19, "L"): 28, (19, "M"): 26, (19, "Q"): 26, (19, "H"): 26,
                  (20, "L"): 28, (20, "M"): 26, (20, "Q"): 30, (20, "H"): 28,
                  (21, "L"): 28, (21, "M"): 26, (21, "Q"): 28, (21, "H"): 30,
                  (22, "L"): 28, (22, "M"): 28, (22, "Q"): 30, (22, "H"): 24,
                  (23, "L"): 30, (23, "M"): 28, (23, "Q"): 30, (23, "H"): 30,
                  (24, "L"): 30, (24, "M"): 28, (24, "Q"): 30, (24, "H"): 30,
                  (25, "L"): 26, (25, "M"): 28, (25, "Q"): 30, (25, "H"): 30,
                  (26, "L"): 28, (26, "M"): 28, (26, "Q"): 28, (26, "H"): 30,
                  (27, "L"): 30, (27, "M"): 28, (27, "Q"): 30, (27, "H"): 30,
                  (28, "L"): 30, (28, "M"): 28, (28, "Q"): 30, (28, "H"): 30,
                  (29, "L"): 30, (29, "M"): 28, (29, "Q"): 30, (29, "H"): 30,
                  (30, "L"): 30, (30, "M"): 28, (30, "Q"): 30, (30, "H"): 30,
                  (31, "L"): 30, (31, "M"): 28, (31, "Q"): 30, (31, "H"): 30,
                  (32, "L"): 30, (32, "M"): 28, (32, "Q"): 30, (32, "H"): 30,
                  (33, "L"): 30, (33, "M"): 28, (33, "Q"): 30, (33, "H"): 30,
                  (34, "L"): 30, (34, "M"): 28, (34, "Q"): 30, (34, "H"): 30,
                  (35, "L"): 30, (35, "M"): 28, (35, "Q"): 30, (35, "H"): 30,
                  (36, "L"): 30, (36, "M"): 28, (36, "Q"): 30, (36, "H"): 30,
                  (37, "L"): 30, (37, "M"): 28, (37, "Q"): 30, (37, "H"): 30,
                  (38, "L"): 30, (38, "M"): 28, (38, "Q"): 30, (38, "H"): 30,
                  (39, "L"): 30, (39, "M"): 28, (39, "Q"): 30, (39, "H"): 30,
                  (40, "L"): 30, (40, "M"): 28, (40, "Q"): 30, (40, "H"): 30,
                  }

# 版本+容錯的 Group1中Blocks的的數量; Group1的Block中data codeword數量; Group2中Blocks的數量; Group2的Block中data codeword數量
BLOCK_INFO = {(1, "L"): (1, 19, 0, 0), (1, "M"): (1, 16, 0, 0), (1, "Q"): (1, 13, 0, 0), (1, "H"): (1, 9, 0, 0),
              (2, "L"): (1, 34, 0, 0), (2, "M"): (1, 28, 0, 0), (2, "Q"): (1, 22, 0, 0), (2, "H"): (1, 16, 0, 0),
              (3, "L"): (1, 55, 0, 0), (3, "M"): (1, 44, 0, 0), (3, "Q"): (2, 17, 0, 0), (3, "H"): (2, 13, 0, 0),
              (4, "L"): (1, 80, 0, 0), (4, "M"): (2, 32, 0, 0), (4, "Q"): (2, 24, 0, 0), (4, "H"): (4, 9, 0, 0),
              (5, "L"): (1, 108, 0, 0), (5, "M"): (2, 43, 0, 0), (5, "Q"): (2, 15, 2, 16), (5, "H"): (2, 11, 2, 12),
              (6, "L"): (2, 68, 0, 0), (6, "M"): (4, 27, 0, 0), (6, "Q"): (4, 19, 0, 0), (6, "H"): (4, 15, 0, 0),
              (7, "L"): (2, 78, 0, 0), (7, "M"): (4, 31, 0, 0), (7, "Q"): (2, 14, 4, 15), (7, "H"): (4, 13, 1, 14),
              (8, "L"): (2, 97, 0, 0), (8, "M"): (2, 38, 2, 39), (8, "Q"): (4, 18, 2, 19), (8, "H"): (4, 14, 2, 15),
              (9, "L"): (2, 116, 0, 0), (9, "M"): (3, 36, 2, 37), (9, "Q"): (4, 16, 4, 17), (9, "H"): (4, 12, 4, 13),
              (10, "L"): (2, 68, 2, 69), (10, "M"): (4, 43, 1, 44), (10, "Q"): (6, 19, 2, 20), (10, "H"): (6, 15, 2, 16),
              (11, "L"): (4, 81, 0, 0), (11, "M"): (1, 50, 4, 51), (11, "Q"): (4, 22, 4, 23), (11, "H"): (3, 12, 8, 13),
              (12, "L"): (2, 92, 2, 93), (12, "M"): (6, 36, 2, 37), (12, "Q"): (4, 20, 6, 21), (12, "H"): (7, 14, 4, 15),
              (13, "L"): (4, 107, 0, 0), (13, "M"): (8, 37, 1, 38), (13, "Q"): (8, 20, 4, 21), (13, "H"): (12, 11, 4, 12),
              (14, "L"): (3, 115, 1, 116), (14, "M"): (4, 40, 5, 41), (14, "Q"): (11, 16, 5, 17), (14, "H"): (11, 12, 5, 13),
              (15, "L"): (5, 87, 1, 88), (15, "M"): (5, 41, 5, 42), (15, "Q"): (5, 24, 7, 25), (15, "H"): (11, 12, 7, 13),
              (16, "L"): (5, 98, 1, 99), (16, "M"): (7, 45, 3, 46), (16, "Q"): (15, 19, 2, 20), (16, "H"): (3, 15, 13, 16),
              (17, "L"): (1, 107, 5, 108), (17, "M"): (10, 46, 1, 47), (17, "Q"): (1, 22, 15, 23), (17, "H"): (2, 14, 17, 15),
              (18, "L"): (5, 120, 1, 121), (18, "M"): (9, 43, 4, 44), (18, "Q"): (17, 22, 1, 23), (18, "H"): (2, 14, 19, 15),
              (19, "L"): (3, 113, 4, 114), (19, "M"): (3, 44, 11, 45), (19, "Q"): (17, 21, 4, 22), (19, "H"): (9, 13, 16, 14),
              (20, "L"): (3, 107, 5, 108), (20, "M"): (3, 41, 13, 42), (20, "Q"): (15, 24, 5, 25), (20, "H"): (15, 15, 10, 16),
              (21, "L"): (4, 116, 4, 117), (21, "M"): (17, 42, 0, 0), (21, "Q"): (17, 22, 6, 23), (21, "H"): (19, 16, 6, 17),
              (22, "L"): (2, 111, 7, 112), (22, "M"): (17, 46, 0, 0), (22, "Q"): (7, 24, 16, 25), (22, "H"): (34, 13, 0, 0),
              (23, "L"): (4, 121, 5, 122), (23, "M"): (4, 47, 14, 48), (23, "Q"): (11, 24, 14, 25), (23, "H"): (16, 15, 14, 16),
              (24, "L"): (6, 117, 4, 118), (24, "M"): (6, 45, 14, 46), (24, "Q"): (11, 24, 16, 25), (24, "H"): (30, 16, 2, 17),
              (25, "L"): (8, 106, 4, 107), (25, "M"): (8, 47, 13, 48), (25, "Q"): (7, 24, 22, 25), (25, "H"): (22, 15, 13, 16),
              (26, "L"): (10, 114, 2, 115), (26, "M"): (19, 46, 4, 47), (26, "Q"): (28, 22, 6, 23), (26, "H"): (33, 16, 4, 17),
              (27, "L"): (8, 122, 4, 123), (27, "M"): (22, 45, 3, 46), (27, "Q"): (8, 23, 26, 24), (27, "H"): (12, 15, 28, 16),
              (28, "L"): (3, 117, 10, 118), (28, "M"): (3, 45, 23, 46), (28, "Q"): (4, 24, 31, 25), (28, "H"): (11, 15, 31, 16),
              (29, "L"): (7, 116, 7, 117), (29, "M"): (21, 45, 7, 46), (29, "Q"): (1, 23, 37, 24), (29, "H"): (19, 15, 26, 16),
              (30, "L"): (5, 115, 10, 116), (30, "M"): (19, 47, 10, 48), (30, "Q"): (15, 24, 25, 25), (30, "H"): (23, 15, 25, 16),
              (31, "L"): (13, 115, 3, 116), (31, "M"): (2, 46, 29, 47), (31, "Q"): (42, 24, 1, 25), (31, "H"): (23, 15, 28, 16),
              (32, "L"): (17, 115, 0, 0), (32, "M"): (10, 46, 23, 47), (32, "Q"): (10, 24, 35, 25), (32, "H"): (19, 15, 35, 16),
              (33, "L"): (17, 115, 1, 116), (33, "M"): (14, 46, 21, 47), (33, "Q"): (29, 24, 19, 25), (33, "H"): (11, 15, 46, 16),
              (34, "L"): (13, 115, 6, 116), (34, "M"): (14, 46, 23, 47), (34, "Q"): (44, 24, 7, 25), (34, "H"): (59, 16, 1, 17),
              (35, "L"): (12, 121, 7, 122), (35, "M"): (12, 47, 26, 48), (35, "Q"): (39, 24, 14, 25), (35, "H"): (22, 15, 41, 16),
              (36, "L"): (6, 121, 14, 122), (36, "M"): (6, 47, 34, 48), (36, "Q"): (46, 24, 10, 25), (36, "H"): (2, 15, 64, 16),
              (37, "L"): (17, 122, 4, 123), (37, "M"): (29, 46, 14, 47), (37, "Q"): (49, 24, 10, 25), (37, "H"): (24, 15, 46, 16),
              (38, "L"): (4, 122, 18, 123), (38, "M"): (13, 46, 32, 47), (38, "Q"): (48, 24, 14, 25), (38, "H"): (42, 15, 32, 16),
              (39, "L"): (20, 117, 4, 118), (39, "M"): (40, 47, 7, 48), (39, "Q"): (43, 24, 22, 25), (39, "H"): (10, 15, 67, 16),
              (40, "L"): (19, 118, 6, 119), (40, "M"): (18, 47, 31, 48), (40, "Q"): (34, 24, 34, 25), (40, "H"): (20, 15, 61, 16),
              }


class QrArgs():
    def __init__(self, version, level, mask):
        self._version = QrVersion(version)
    
        # 以下兩個參數較無額外的處理
        # 因此直接assign，不利用額外的Class處理了
        assert level in ["L", "M", "Q", "H"], "level must be LMQH!"
        self._level = level
        assert 0 <= mask <= 7, "mask must between 0 and 7!"
        self._mask = mask

    # ===============================
    # 以下為QrVersion內所定義的attribute
    # ===============================
    
    @property
    def version(self):
        return self._version.number

    @property
    def size(self):
        return self._version.size

    @property
    def possible_center(self):
        return self._version.possible_center

    @property
    def version_info(self):
        return self._version.version_info

    @property
    def remainder_bit(self):
        return self._version.remainder_bit

    
    # level
    @property
    def level(self):
        return self._level

    # mask
    @property
    def mask(self):
        return self._mask

    # ======================================
    # 以下為經由version, level所定義的attribute
    # ======================================
    @property
    def data_capacity(self):
        return DATA_CAPACITY[(self.version, self.level)]

    @property
    def neceb(self):
        return ERROR_CODEWORD[(self.version, self.level)]

    @property
    def block_info(self):
        return BLOCK_INFO[(self.version, self.level)]

    @property
    def nbg1(self):
        return self.block_info[0]

    @property
    def ndcg1b(self):
        return self.block_info[1]

    @property
    def nbg1b(self):
        return self.ndcg1b * 8

    @property
    def nbg2(self):
        return self.block_info[2]

    @property
    def ndcg2b(self):
        return self.block_info[3]

    @property
    def nbg2b(self):
        return self.ndcg2b * 8

    @property
    def dpppe(self):
        """
        data plus padding plus error的bits數
        """
        return self.data_capacity + (self.nbg1 + self.nbg2) * self.neceb * 8
