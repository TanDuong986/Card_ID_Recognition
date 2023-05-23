# CS_Final_Project
* This is final project of `Cognitive Science`, we're going to build a model related to text recognition use for digitizing citizen information. 
* The project have two parts is AI model and Theme to show result. Waiting for us ... :rocket: :rocket: :rocket:
* The member of this project: [dtan986](https://www.linkedin.com/in/tan-duong-622189225/) and [anh147](https://www.linkedin.com/in/anh147/)

you can get more detail in [here](./AI_corner/Code)

Run GUI:
* Step 1: Install Tesseract at [link](https://github.com/UB-Mannheim/tesseract/wiki)
* Step 2: 
cd Theme  \\ python gui.py

---
Structure of this repo:
```
main
|__AI_corner
|          |__Code
|          |__Data_OCR
|
|___Theme
        |___output
        |___pths
        |___result
```
---
main file for inference independently not use GUI is [inference code](./AI_corner/Code/inference.py)
you can try:
```
cd ./AI_corner/Code
python inference.py --img /img_valid/thuThi.jpg
```
If it not run, feel free to install missing dependencies. Contact me if you have any question: [dtan986](duongtanrb@gmai.com)

