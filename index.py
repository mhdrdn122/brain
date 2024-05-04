from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def decodeImage () :
    image_data = request.form['image_data'].split(',')[1]
    nparr = np.fromstring(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def encodeImage(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def mask_white(img , lower, upper):
    img_test = img
    # تحويل الصورة إلى الفضاء اللوني HSV
    hsv_image = cv2.cvtColor(img_test, cv2.COLOR_BGR2HSV)
    # تحديد نطاق اللون الأبيض في الفضاء اللوني HSV
    lower_white = np.array(lower)
    upper_white = np.array(upper)
    # إنشاء قناع (mask) للون الأبيض
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # استخدام القناع لعزل المنطقة التي تحتوي على اللون الأبيض
    white_area = cv2.bitwise_and(img_test, img_test, mask=white_mask)
    return [white_area ,white_mask]


@app.route('/filter', methods=['POST'])
def apply_filter():
    img = decodeImage()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_filtered = cv2.bilateralFilter(gray,25 , 75 , 75)

    encoder = encodeImage(img_filtered)
    return encoder


@app.route('/threshold', methods=['POST'])
def apply_threshold():
    img = decodeImage()

    img_thresh = mask_white(img , [0, 0, 200] ,[180, 255, 255])

    encoder = encodeImage(img_thresh[0])
    return encoder

def contours_rect (img , mask):
    img_test = img
    # الحصول على معلومات ال contoure
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # رسم مستطيل أحمر حول كل contoure
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_test, (x, y), (x + w, y + h), (0, 0, 255), 1)
    return img_test

@app.route('/rect', methods=['POST'])
def apply_rect():
    img = decodeImage()

    img_thresh = mask_white(img , [0, 0, 200] ,[180, 255, 255])
    img_rect = contours_rect(img_thresh[0] , img_thresh[1])
    print("test")
    encoder = encodeImage(img_rect)
    return encoder

# def cal_area(contours):
#     # حساب المساحة والقطر لكل contoure
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         diameter = np.sqrt(4 * area / np.pi)
#         if (area != 0) | (diameter != 0):
#             print("Area :", area/10000 , " mm sq")
#             print("Diameter", diameter/100 , "mm")

def contours_draw(img , mask):
    img_test = img
    # الحصول على معلومات ال contoure
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cal_area(contours)
    # رسم حواف المنطقة البيضاء بخط أحمر بسماكة 1 بكسل
    image = cv2.drawContours(img_test, contours, -1, (0, 0, 255), 2)
    return image

@app.route('/border', methods=['POST'])
def apply_border():
    img = decodeImage()

    img_thresh = mask_white(img , [0, 0, 200] ,[180, 255, 255])
    img_border = contours_draw(img_thresh[0] , img_thresh[1])

    encoder = encodeImage(img_border)
    return encoder

if __name__ == '__main__':
    app.run(debug=True)
