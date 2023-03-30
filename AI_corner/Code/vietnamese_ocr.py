import pytesseract
import argparse
import cv2
import os
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path of image file")
ap.add_argument("-p", "--preprocess",type=str,default="thresh",help="Preprocessing")
args= vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif args["preprocess"] == "blur":
    gray = cv2.medianBlur(gray,3)

filename = "AI_corner\\Code\\Output\\{}.jpg".format(os.getpid())
cv2.imwrite(filename,gray)

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

text = pytesseract.image_to_string(Image.open(filename),lang='vie')

os.remove(filename)

print(text)

cv2.imshow("image",gray)
cv2.waitKey(0)