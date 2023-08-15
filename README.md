# IS-QRcode
實作碩士論文中，利用BlendMask實例分割模型與高斯-約旦消去法，將QR code與背景圖片做融合，產生有視覺意義的QR code。

根據ISO 18004:2015的規範，QR code定義了不同的版本(level)、容錯等級(correction level)。

根據版本、容錯等級、加入的訊息，遵循QR code的編碼方式，產生最原始、不具視覺意義的QR code。(範例中，版本：5、容錯等級：L、訊息：Aesthetic QR)

<img src="https://github.com/steven520610/IS-QRcode/assets/49943356/165aee33-0df0-4547-b7a4-66c405ae0bf1" width="200px">

依據傳入的圖片路徑參數，將要加入到QR code內的背景圖片，使用OTSU方法，轉換成二值化影像。

<img src="https://github.com/steven520610/IS-QRcode/assets/49943356/e64ab43f-d90f-454d-99ac-d146d7539068" width="200px"> ->
<img src="https://github.com/steven520610/IS-QRcode/assets/49943356/5cd620f9-3cb7-4353-8579-61ce3b8f8671" width="200px"> 

將二值化影像，依據碼元大小，建立一個基於碼元的二值化影像。

<img src="https://github.com/steven520610/IS-QRcode/assets/49943356/5cd620f9-3cb7-4353-8579-61ce3b8f8671" width="200px"> ->
<img src="https://github.com/steven520610/IS-QRcode/assets/49943356/d135aa94-ca25-4d32-9a0a-ad756a8ab504" width="200px">

利用BlendMask，將背景圖片的ROI(Region of Interest)取出來。

<img src="https://github.com/steven520610/IS-QRcode/assets/49943356/3685f66b-6817-41d1-a9b4-671b9424aca1" width="200px">

把前面的原始QR code、基於碼元的二值化影像、ROI利用高斯-約旦消去法，將三者結合在一起，產生調整過碼元且看的出圖像輪廓，並且能成功掃描的QR code。

<img src="https://github.com/steven520610/IS-QRcode/assets/49943356/6ad565d9-dfc3-4ec9-8115-47e4fcfb52a4" width="200px">

將處理過後的QR code與背景圖片做融合，產生第一步美化過後的QR code。

<img src="https://github.com/steven520610/IS-QRcode/assets/49943356/bd0383ef-db3a-4f54-a3e0-ac398c3067b4" width="200px">

最後根據QR code的容錯機制與解碼器對於影像的解碼方式，在保有解碼的能力下，刪除一部分碼元，完成美化過後的QR code。

<img src="https://github.com/steven520610/IS-QRcode/assets/49943356/65d13340-4650-43c4-a763-b83e67666b8a" width="200px">

使用方法，執行main.py檔案，並且傳入適當的參數。

python3 main.py v l m o img --mask -ms -s

v: QR code所使用的版本(1~40)。

l: QR code所使用的容錯等級(L, M, Q, H)。

m: 要加入QR code的訊息。

img: 使用的背景圖片之路徑。

o: 輸出圖片要存放之資料夾。

mask: 建立QR code時所使用的mask(0~7)。

ms: 碼元的大小，即每個碼元內會存在多少個pixels。
s: 碼元的切分大小。

參考資料

https://yeecy.medium.com/%E5%A6%82%E4%BD%95%E8%A3%BD%E4%BD%9C-qr-code-0-%E5%89%8D%E8%A8%80-e464466dc321

https://www.thonky.com/qr-code-tutorial/

https://research.swtch.com/qart
