import pytesseract
import argparse
import cv2
import os
from PIL import Image
import numpy as np

def read(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,3)
    gray = cv2.GaussianBlur(gray,(1,1),0)
    gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    export_fol = './output'
    if not os.path.exists(export_fol):
        os.makedirs(export_fol)
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    instance = Image.fromarray(cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB))
    text = pytesseract.image_to_string(instance,lang='vie')
    return text