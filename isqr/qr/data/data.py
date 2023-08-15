# Date: 2023.8.15
# Author: Steven
"""
根據QR code的參數(level, level)
用來處理有關訊息的class
包括將原始訊息編碼成bit stream
將bit stream轉化成RS code
RS code之間做XOR運算等等。
"""

from ..args import QrArgs

# 利用別人已經寫好，用來處理RS code的套件
# 將訊息編碼成RS code。
import unireedsolomon as rs
from copy import deepcopy
# 新增各個字元的ASCII編碼
ALPHANUMERIC_CHAR = {}
# 0~9
for i in range(10):
    ALPHANUMERIC_CHAR[str(i)] = i
# A~Z
for i in range(26):
    ALPHANUMERIC_CHAR[chr(65+i)] = i+10
# other specific char
special_char = " $%*+-./:"
for i in range(len(special_char)):
    ALPHANUMERIC_CHAR[special_char[i]] = i+36


class QrData():
    def __init__(self, args: QrArgs, message):
        """Class that process input message

        Args:
            args (QrArgs)
            messgae (str): input message from cmd
        """
        assert isinstance(args, QrArgs), "args must be QrArgs instance"
        assert isinstance(message, str), "Input message is not a string!"
        self._args = args
        self._message = message
        # module count every block: 儲存每個block(block1, block2)中所存的module數量
        self.mceb = [0] * (args.nbg1 + args.nbg2)

    @property
    def message(self):
        return self._message

    @property
    def message_length(self):
        return len(self.message)

    def decide_encode_mode(self) -> int:
        """
        根據訊息，決定此訊息要使用QR code內所使用的哪種編碼模式
        此處只定義numeric, alphanumeric, byte三種模式
        缺少漢字模式。
        Returns:
            int: 0 for numeric
                 1 for alphanumeric
                 2 for byte
        """
        # numeric
        if self._message.isdigit():
            self._mode = 0
            print("Encode mode: numeric")
            return
        # byte
        for char in self.message:
            if char not in ALPHANUMERIC_CHAR.keys():
                self._mode = 2
                print("Encode mode: byte")
                return
        # alphanumeric
        self._mode = 1
        print("Encode mode: alphanumeric")
        return

    @property
    def mode(self):
        return self._mode

    # =============
    # 將訊息轉化成位元
    # =============
    def message2bitstream(self):
        # Initialize
        self._bitstream = ""
        message = self.message

        # numeric
        if self.mode == 0:
            # mode indicator
            self._bitstream += "0001"

            # count indicator
            if 1 <= self._args.version <= 9:
                self._bitstream += bin(self.message_length)[2:].rjust(10, "0")
            elif 10 <= self._args.version <= 26:
                self._bitstream += bin(self.message_length)[2:].rjust(12, "0")
            else:
                self._bitstream += bin(self.message_length)[2:].rjust(14, "0")

            # encode
            while message:
                if len(message) >= 3:
                    self._bitstream += bin(int(message[:3]))[2:].rjust(10, "0")
                elif len(message) == 2:
                    self._bitstream += bin(int(message[:3]))[2:].rjust(7, "0")
                else:
                    self._bitstream += bin(int(message[:3]))[2:].rjust(4, "0")
                message = message[3:]

        # alphanumeric
        elif self.mode == 1:
            # mode indicator
            self._bitstream += "0010"

            # count indicator
            if 1 <= self._args.version <= 9:
                self._bitstream += bin(self.message_length)[2:].rjust(9, "0")
            elif 10 <= self._args.version <= 26:
                self._bitstream += bin(self.message_length)[2:].rjust(11, "0")
            else:
                self._bitstream += bin(self.message_length)[2:].rjust(13, "0")

            # encode
            while message:
                if len(message) >= 2:
                    self._bitstream += bin(int(ALPHANUMERIC_CHAR[message[0]]
                                               * 45 + ALPHANUMERIC_CHAR[message[1]]))[2:].rjust(11, "0")
                else:
                    self._bitstream += bin(
                        int(ALPHANUMERIC_CHAR[message[0]]))[2:].rjust(6, "0")
                message = message[2:]

        # byte
        else:
            # mode indicator
            self._bitstream += "0100"

            # count indicator
            if 1 <= self._args.version <= 9:
                self._bitstream += bin(self.message_length)[2:].rjust(8, "0")
            elif 10 <= self._args.version <= 26:
                self._bitstream += bin(self.message_length)[2:].rjust(10, "0")
            else:
                self._bitstream += bin(self.message_length)[2:].rjust(10, "0")

            # encode
            for char in message:
                self._bitstream += bin(ord(char))[2:].rjust(8, "0")

        # 在Terminator前面加上判斷式
        # 如果encode完後，data的長度超過版本+容錯下規定的長度
        # 則報錯
        assert self.bitstream_length <= self._args.data_capacity, "The message is too long to encode!"

        # Terminator
        # 訊息夠長，只差小於4個bit即可滿足的情況
        if self._args.data_capacity - self.bitstream_length < 4:
            self._bitstream = self._bitstream.ljust(
                self._args.data_capacity, "0")
        else:
            self._bitstream += "0000"

        # 先判斷是否已經符合規定的長度
        # 如果符合就直接回傳bit_stream了，不需要再做後面的處理
        # 否則後面while loop時會造成無窮迴圈
        if self.bitstream_length == self._args.data_capacity:
            return

        # ＊＊若補完Terminator長度仍不到8的倍數，則繼續補0
        if self.bitstream_length % 8:
            self._bitstream = self._bitstream.ljust(
                8*(self.noc+1), "0")
        self._data_bitstream = self.bitstream

        # Padding
        flag = 1
        padding_type_1 = "11101100"
        padding_type_2 = "00010001"
        self._nopc = 0

        while self.bitstream_length != self._args.data_capacity:
            if flag:
                self._bitstream += padding_type_1
                self._nopc += 1
                flag = 0
            else:
                self._bitstream += padding_type_2
                self._nopc += 1
                flag = 1

    # ====================================
    # 以下定義有關bit stream以及codeword的屬性
    # ====================================
    @property
    def bitstream(self):
        return self._bitstream

    @property
    def bitstream_length(self):
        return len(self.bitstream)

    @property
    def noc(self):
        """
        Number of codewords
        """
        return self.bitstream_length // 8

    @property
    def data_bitstream(self):
        return self._data_bitstream

    @property
    def data_bitstream_length(self):
        return len(self.data_bitstream)

    @property
    def nodc(self):
        """
        Number of data codewords
        """
        return self.data_bitstream_length // 8

    @property
    def nopc(self):
        """
        Number of padding codewords
        """
        return self._nopc

    # ======================
    # 將訊息轉換成的bit stream
    # 利用特殊的編碼器
    # 轉換成RS code。
    # ======================
    def encode(self):
        num_of_codeword = self._args.data_capacity // 8
        # 把位元轉換成十進制，以符合RSCode package的需求
        message_list = []
        for i in range(num_of_codeword):
            num = int(self.bitstream[8*i:8*(i+1)], 2)
            message_list.append(num)
            
        # Interleaving
        # 宣告要存放data_codeword, type_codeword, error_codeword的Group, Block
        self.data_codeword_lst = []
        self.type_codeword_lst = []
        self.error_codeword_lst = []

        # 使用index而不是在loop內用i
        # 是因為不同的block在執行loop時要使用相同的index
        # 如果每次loop index都從頭開始的話
        # 會取到錯誤的資料
        index = 0

        # Group 1
        for _ in range(self._args.nbg1):
            # 每次處理一個block時，都重新宣告一個空的陣列，用來存每個block的資料
            sub_lst = []
            # 建立判斷每個codeword屬於哪一種(message, padding, parity)的sub_lst
            # 1:message, 2:padding, 3:parity
            type_sub_lst = []
            for _ in range(self._args.ndcg1b):
                # 依據規定的version, level，決定每個block要存放多少筆資料
                sub_lst.append(message_list[index])
                if index+1 <= self.nodc:
                    type_sub_lst.append(1)
                else:
                    type_sub_lst.append(2)
                index += 1

            # 一個block的資料存完之後，因為可能還有更多block(同一個Group內)
            # 所以把當前存完資料的陣列，append到最初宣告的data_codeword_lst陣列，形成一個二維陣列
            # dimension 1:Group內的第幾個block
            # dimension 2:block內的資料
            self.data_codeword_lst.append(sub_lst)
            self.type_codeword_lst.append(type_sub_lst)

            # 因為每組的n, k可能都不一樣，所以每次都要重新宣告
            # 而且產生error_codeword是依據每一個block去產生，並不是用全部的資料(所以info的表原本是錯的，不用乘起來相加才對)
            # n:data_codeword + error_codeword的個數
            # k:data_codeword的個數
            # generator: alpha
            # fcr:設為0
            # prim:byte_wise modulo的值，在QRcode為100011101 >> 285 (即本原多項式)
            # c_exp: QRcode在RS code內使用的Galois Field為 GF(2^8)，所以設定8
            RS_encoder = rs.rs.RSCoder(
                n=self._args.ndcg1b + self._args.neceb,
                k=self._args.ndcg1b,
                generator=2,
                prim=285,
                fcr=0,
                c_exp=8)
            
            
            # 因為會依據每個block處理，所以此處傳入的為sub_lst(1D陣列)
            # 回傳一個長度為n且資料為binary的1D陣列
            error_codeword = RS_encoder.encode(sub_lst, return_string=False)
            # 把lst內的每個binary data轉回integer的形式
            # 原本為GF2int的形式
            error_codeword = [int(num) for num in error_codeword]
            # 前面做的結果是長度為n且包含原始的message(RS code的特性)的陣列，這邊把只屬於更正碼的部分取出來
            error_codeword = error_codeword[self._args.ndcg1b:]
            # 最後的error_codeword_lst是一個2D的陣列
            # dimension 1:Group內的第幾個block
            # dimension 2:block內的error correction codeword的資料
            self.error_codeword_lst.append(error_codeword)

        # Group 2 的處理，基本上和Group 1的處理方法一模一樣
        for _ in range(self._args.nbg2):
            sub_lst = []
            type_sub_lst = []
            for _ in range(self._args.ndcg2b):
                sub_lst.append(message_list[index])
                if index+1 <= self.nodc:
                    type_sub_lst.append(1)
                else:
                    type_sub_lst.append(2)
                index += 1
            self.data_codeword_lst.append(sub_lst)
            self.type_codeword_lst.append(type_sub_lst)

            RS_encoder = rs.rs.RSCoder(
                n=self._args.ndcg2b + self._args.neceb,
                k=self._args.ndcg2b,
                generator=2,
                prim=285,
                fcr=0,
                c_exp=8)
            error_codeword = RS_encoder.encode(sub_lst, return_string=False)
            error_codeword = [int(num) for num in error_codeword]
            error_codeword = error_codeword[self._args.ndcg2b:]
            self.error_codeword_lst.append(error_codeword)

    def rearrange_bitstream_with_type(self):
        """
        
        """
        
        # 這邊要用copy複製一個和輸入的data, error codeword list相同的陣列
        # 如果直接用輸入的list的話
        # 經過此function處理過後，輸入的list都會因為此function內的修改
        # 變成空的list
        # 然而我們後面還需要使用
        # 所以要複製一個指向不同記憶體空間，但是值是相同的list
        d_list = deepcopy(self.data_codeword_lst)
        e_list = deepcopy(self.error_codeword_lst)

        final_message = []

        # block for each module
        self._bfem = ""
        self._type_codeword = []
        iter = max(self._args.ndcg1b, self._args.ndcg2b)
        for _ in range(iter):
            # example為(2, 15, 2, 16)
            # 所以此處range為4
            for i in range(self._args.nbg1 + self._args.nbg2):
                if d_list[i]:
                    # 每次取都把該block的第一個data取出來之後
                    # 再把該block重新assign成剩下的資料構成的陣列
                    final_message.append(d_list[i][0])
                    d_list[i] = d_list[i][1:]
                    self._bfem += str(i+1) * 8

                if self.type_codeword_lst[i]:
                    self.type_codeword.append(self.type_codeword_lst[i][0])
                    self.type_codeword_lst[i] = self.type_codeword_lst[i][1:]
        # 再放error_codeword
        for _ in range(self._args.neceb):
            for i in range(self._args.nbg1 + self._args.nbg2):
                final_message.append(e_list[i][0])
                e_list[i] = e_list[i][1:]

                self._bfem += str(i+1) * 8

                self._type_codeword.append(3)

        # 最後再把十進位的message轉換成二進位
        self.interleaved_bitstream = ""
        for num in final_message:
            self.interleaved_bitstream += bin(num)[2:].rjust(8, "0")

    def add_remainder(self):
        self.interleaved_bitstream += "0" * self._args._version.remainder_bit

    @property
    def bfem(self):
        return self._bfem

    @property
    def type_codeword(self):
        return self._type_codeword

    # 做RS_encoding時，應該要一個block一個block去做
    # 除非只有一個block，否則不能一整串bit來處理

    @classmethod
    def extra_RS_encoding(cls, bit_stream, n, k):
        # 因應rs.rs.RSCoder的要求，要把bit stream
        # 轉成integer的array才可以傳入
        def bit_stream_to_codeword(bit_stream):
            codeword_value_list = []
            for i in range(len(bit_stream) // 8):
                codeword_value_list.append(int(bit_stream[i * 8:(i+1)*8], 2))
            return codeword_value_list

        codeword_value_list = bit_stream_to_codeword(bit_stream)
        RS_encoder = rs.rs.RSCoder(
            n=n, k=k, generator=2, prim=285, fcr=0, c_exp=8)
        error_codeword = RS_encoder.encode(
            codeword_value_list, return_string=False)
        error_codeword = [bin(int(num))[2:].rjust(8, "0")
                          for num in error_codeword]
        bit_stream = ""
        for byte in error_codeword:
            bit_stream += byte
        return bit_stream

    @classmethod
    def XOR(cls, codeword_list1, bit2):
        bit1 = ""
        XOR_list = []
        for i in range(len(codeword_list1)):
            bit1 += bin(codeword_list1[i])[2:].rjust(8, "0")

        int_bit1, int_bit2 = int(bit1, 2), int(bit2, 2)
        int_XOR = int_bit1 ^ int_bit2
        bit_XOR = bin(int_XOR)[2:].rjust(len(bit1), "0")
        for i in range(len(bit1) // 8):
            XOR_list.append(int(bit_XOR[i * 8:(i+1) * 8], 2))
        return XOR_list

    def rearrange_bitstream_without_type(self):

        d_list = self.data_codeword_lst.copy()
        e_list = self.error_codeword_lst.copy()

        final_message = []
        iter = max(self._args.ndcg1b, self._args.ndcg2b)
        for _ in range(iter):
            # example為(2, 15, 2, 16)
            # 所以此處range為4
            for i in range(self._args.nbg1 + self._args.nbg2):
                if d_list[i]:
                    # 每次取都把該block的第一個data取出來之後
                    # 再把該block重新assign成剩下的資料構成的陣列
                    final_message.append(d_list[i][0])
                    d_list[i] = d_list[i][1:]

        # 再放error_codeword
        for _ in range(self._args.neceb):
            for i in range(self._args.nbg1 + self._args.nbg2):
                final_message.append(e_list[i][0])
                e_list[i] = e_list[i][1:]

        # 最後再把十進位的message轉換成二進位
        self.interleaved_bitstream = ""
        for num in final_message:
            self.interleaved_bitstream += bin(num)[2:].rjust(8, "0")
