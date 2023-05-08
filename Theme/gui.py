# Welcome to PyShine
import pytesseract
import cv2, os,sys
from PIL import Image
import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore,QtGui,QtWidgets
import glob

# Here we will get the path of the tessdata
# For 64 bit installation of tesseract OCR 
language_path = 'C:\\Program Files\\Tesseract-OCR\\tessdata\\'
language_path_list = glob.glob(language_path+"*.traineddata")



language_names_list = []

for path in language_path_list:
	base_name =  os.path.basename(path)
	base_name = os.path.splitext(base_name)[0]
	language_names_list.append(base_name)



font_list = []
font = 2

for font in range(110):
	font+=2
	font_list.append(str(font))

# print('Font list:',font_list)

class PyShine_OCR_APP(QtWidgets.QMainWindow):
	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		self.ui = uic.loadUi('main.ui',self)
		self.image = None
		
		self.ui.pushButton.clicked.connect(self.open)
		self.rubberBand = QRubberBand(QRubberBand.Rectangle,self)
		self.ui.label_2.setMouseTracking(True)
		self.ui.label_2.installEventFilter(self)
		self.ui.label_2.setAlignment(PyQt5.QtCore.Qt.AlignTop)
		
		self.language = 'eng'
		self.comboBox.addItems(language_names_list)
		self.comboBox.currentIndexChanged['QString'].connect(self.update_now)
		self.comboBox.setCurrentIndex(language_names_list.index(self.language))
		
		self.font_size = '20'
		self.text = ''
		self.comboBox_2.addItems(font_list)
		self.comboBox_2.currentIndexChanged['QString'].connect(self.update_font_size)
		self.comboBox_2.setCurrentIndex(font_list.index(self.font_size))
		
		self.ui.textEdit.setFontPointSize(int(self.font_size))
		self.setAcceptDrops(True)
		self.ui.pushButton_2.clicked.connect(self.copytext)
		
	def update_now(self,value):
		self.language = value
		print('Language Selected as:',self.language)
	
	def update_font_size(self,value):
		self.font_size = value
		self.ui.textEdit.setFontPointSize(int(self.font_size))
		self.ui.textEdit.setText(str(self.text))
	
	
	def open(self):
		filename = QFileDialog.getOpenFileName(self,'Select File')
		self.image = cv2.imread(str(filename[0]))
		frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
		image =  QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
		self.ui.label_2.setPixmap(QPixmap.fromImage(image))
	
	def copytext(self):
		# self.ui.textEdit.copy()
		# print("Copy text", self.ui.textEdit.toPlainText())

		clipboard.setText(self.ui.textEdit.toPlainText())
		msg = QMessageBox()
		msg.setText("Copied to clipboard")
		msg.exec_()
	
	def image_to_text(self,crop_cvimage):
		gray = cv2.cvtColor(crop_cvimage,cv2.COLOR_BGR2GRAY)
		gray = cv2.medianBlur(gray,1)
		crop = Image.fromarray(gray)
		text = pytesseract.image_to_string(crop,lang = self.language)
		print('Text:',text)
		return text
	
	def eventFilter(self,source,event):
		width = 0
		height = 0
		if (event.type() == QEvent.MouseButtonPress and source is self.ui.label_2):
			self.org = self.mapFromGlobal(event.globalPos())
			self.left_top = event.pos()
			self.rubberBand.setGeometry(QRect(self.org,QSize()))
			self.rubberBand.show()
		elif (event.type() == QEvent.MouseMove and source is self.ui.label_2):
			if self.rubberBand.isVisible():
				self.rubberBand.setGeometry(QRect(self.org,self.mapFromGlobal(event.globalPos())).normalized())
		elif(event.type() == QEvent.MouseButtonRelease and source is self.ui.label_2):
			if self.rubberBand.isVisible():
				self.rubberBand.hide()
				rect = self.rubberBand.geometry()
				self.x1 = self.left_top.x()
				self.y1 = self. left_top.y()
				width = rect.width()
				height = rect.height()
				self.x2 = self.x1+ width
				self.y2 = self.y1+ height
			if width >=10 and height >= 10  and self.image is not None:
				self.crop = self.image[self.y1:self.y2, self.x1:self.x2]
				# cv2.imwrite('cropped.png',self.crop)
				self.text = self.image_to_text(self.crop)
				self.ui.textEdit.setText(str(self.text))
			else:
				self.rubberBand.hide()
		else:
			return 0
		return QWidget.eventFilter(self,source,event)
	
# www.pyshine.com
app = QtWidgets.QApplication(sys.argv)
clipboard = app.clipboard()
mainWindow = PyShine_OCR_APP()
mainWindow.show()
sys.exit(app.exec_())

