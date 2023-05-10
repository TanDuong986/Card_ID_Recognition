import pytesseract
import argparse
import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import mmap


from match_word import findM


def pp(img):
    gray = cv2.GaussianBlur(img,(7,7),0)
    # cv2.imwrite("gausian3.jpg",gray)
    gray = cv2.medianBlur(gray,5)
    # cv2.imwrite("median4.jpg",gray)
    gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imwrite("thresh5.jpg",gray)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel=np.ones((5,5),np.uint8))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel=np.ones((3,3),np.uint8))
    # cv2.imwrite("morphology6.jpg",gray)
    # # gray = cv2.erode(gray,iterations=1,kernel=kernel)
    gray = cv2.erode(gray,iterations=1,kernel=np.ones((3,3),np.uint8))
    # cv2.imwrite("erode7.jpg",gray)
    return gray

def textEx(image): # convert Image to raw text
    h,w,_ = image.shape
    image = cv2.resize(image,(int(w*3.1),int(h*3.1)),interpolation=cv2.INTER_AREA)
    # cv2.imwrite("scale1.jpg",image)
    lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB) # l channel
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV_FULL)
    l_channel,a,b = cv2.split(lab)
    h,s,v = cv2.split(hsv) # value channel
 
    ## test on 3 channel______________________________
    # gray1 = pp(image[:,:,1])
    # gray2 = pp(l_channel)
    # gray3 = pp(v)

    # export_fol = './output'
    # if not os.path.exists(export_fol):
    #     os.makedirs(export_fol)

    # pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    # tO = []
    # for _,sample in enumerate([gray1,gray2,gray3]):
    #     filename = os.path.join(export_fol,"tmp.jpg")
    #     cv2.imwrite(filename,sample)
    #     text = pytesseract.image_to_string(Image.open(filename),lang='vie')
    #     tO.append(text)
    #     os.remove(filename)
    
    # Test on 3 channel______________
    key = (0.5*(l_channel) + 0.5*(image[:,:,1])).astype(np.uint8)
    gray = pp(key) # l_channel or v or image[:,:,1]
    plt.imshow(gray,cmap='gray')
    plt.show()
    # cv2.imwrite("lab2.jpg",l_channel)
    export_fol = './output'
    if not os.path.exists(export_fol):
        os.makedirs(export_fol)
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    filename = os.path.join(export_fol,"tmp.jpg")
    cv2.imwrite(filename,gray)
    tO = pytesseract.image_to_string(Image.open(filename),lang='vie')
    os.remove(filename)
    return tO 

def opt():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--image","--img",required=True,help="Path of image file")
    ap.add_argument("-p", "--preprocess",type=str,default="thresh",help="Preprocessing")
    ap.add_argument("--name",type=int,default=0,help="Is this name field")
    args= vars(ap.parse_args())
    return args

def remove_(strr):
    if len(strr) ==0:
        return "None"
    if strr[0] == "," or strr[0] == "." or strr[0] == "@":
        strr = strr[1:]
    if strr[-1] == "," or strr[-1] == "." or strr[-1] == "@":
        strr = strr[:-1]
    return strr 

def filterText(img,isName = False):
    text = textEx(img)
    text = text[:-1]
    text = text.replace("\n"," , ").replace(",", " ,").replace("-"," ")
    idv = text.split(" ")
    idvn = []
    for i in idv:
        if findM(i,isName):
            idvn.append(i)
    outFn = " ".join(idvn)
    return outFn

if __name__ == "__main__":
    file_path = 'general_dict.txt'
    now = time.time()
    mn = 'MG'
    args = opt()
    image = cv2.imread(args["image"])
    outFn = filterText(image,args["name"])
    print(outFn)


        
        






