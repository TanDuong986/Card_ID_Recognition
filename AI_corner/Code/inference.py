import torch
import cv2 
import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import gdown
import math
import matplotlib.pyplot as plt
import argparse
import pytesseract

url = {"last" : "https://drive.google.com/uc?id=1-xrzOsPykarIaV1qAborzmnNbUTRwnoZ", 
       "best" : "https://drive.google.com/uc?id=1-eGX9PkWiEcD6mwKnqWyAmJoSovTh9vP"}

weight = {"last" : "./pths/last.pt",
         "best" : "./pths/best.pt"}

def init_():
    os.makedirs('./pths',exist_ok=True)
    os.makedirs('./results/img',exist_ok=True)
    os.makedirs('./results/info',exist_ok=True) 
    if not os.path.exists('pths/last.pt'):
        gdown.download(url["last"],"pths/last.pt")
    if not os.path.exists('pths/best.pt'):
        gdown.download(url["best"],"pths/best.pt")

def ppTiny(csd): # preprocessing and read text for consider area small
    gray = cv2.cvtColor(csd,cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,3)
    gray = cv2.GaussianBlur(gray,(1,1),0)
    gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    instance = Image.fromarray(cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB))
    text = pytesseract.image_to_string(instance,lang='vie')
    return text


def predict(pth_img,model,threshold = 0.5,sizee=(640,640)): # count number of object per class
    use = Image.open(pth_img).convert('RGB')
    use.resize(sizee)
    detections = model(use)
    results = detections.pandas().xyxy[0].to_dict(orient="records")
    #filter
    arr = []
    for result in results:
        if result['confidence'] >= threshold :
            arr.append(result)
    return sorted(arr, key=lambda x: x["ymin"])

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",type=str,required=True,help="location of image")
    return parser.parse_args()


def revert(pth_img,location):
    img = cv2.imread(pth_img)
    prt = img.copy()  
    use = img.copy()
    cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    color = []

    for i in range(5): # 5 label
        cl = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        color.append(cl)

    dictCl = {'name': color[0], 'home' : color[1], 'add':color[2], 'date':color[3],'id': color[4]}
    field = ['id','name','date','home','add']
    rs = []

    for loc in location:
        x1 = int(loc['xmin'])
        y1 = int(loc['ymin'])
        x2 = int(loc['xmax'])
        y2 = int(loc['ymax'])
        cv2.rectangle(prt,(x1,y1),(x2,y2),color=dictCl[loc['name']],thickness=2)
    cv2.imwrite('./output/result.jpg',prt)

    idx = 0
    i =0
    location.append({'name':'end'})
    for idx in range(len(field)):
        text = ''
        while location[i]['name'] == field[idx]:
            x1 = int(location[i]['xmin'])
            y1 = int(location[i]['ymin'])
            x2 = int(location[i]['xmax'])
            y2 = int(location[i]['ymax'])
            csd = use[y1:y2,x1:x2]
            text += ppTiny(csd)
            cv2.imwrite(f'./output/{field[idx]}_{np.random.randint(1,5)}.jpg',csd)
            i +=1
        rs.append(text)
    
    return rs

        
        
    

if __name__ =="__main__":
    init_()
    args = opt()
    img_path = args.img
    name_img = os.path.basename(img_path)
    out_path = os.path.join("./results/img",name_img) # image out

    model = torch.hub.load('ultralytics/yolov5', 'custom',path=weight["best"], force_reload=True)
    sample = predict(img_path,model) # contain dictionary about location of each object
    rs = revert(img_path,sample) # rs is list of text
    
