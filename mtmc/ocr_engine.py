import re
from paddleocr import PaddleOCR

_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False
        )
    return _ocr

def extract_timestamp(frame):
    ocr = get_ocr()
    h, w = frame.shape[:2]
    crop = frame[0:int(h*0.12), 0:int(w*0.6)]
    
    result = ocr.ocr(crop)

    text = ""
    if result:
        for line in result:
            if line:
                for box in line:
                    text += " " + box[1][0]

    match = re.search(r"\d{2,4}[-/]\d{2}[-/]\d{2,4}\s+\d{2}:\d{2}:\d{2}", text)
    if match:
        return match.group(0)
    return None

def extract_location(frame):
    ocr = get_ocr()
    h, w = frame.shape[:2]
    crop = frame[int(h*0.88):h, int(w*0.35):w]

    result = ocr.ocr(crop)

    text = ""
    if result:
        for line in result:
            if line:
                for box in line:
                    text += " " + box[1][0]

    return text.strip()
