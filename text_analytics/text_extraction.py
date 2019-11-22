# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:10:02 2019

@author: Rajkumar
"""

'''Optical Character Recognition (OCR)'''
import os
import numpy as np
from PIL import Image
import cv2
from pytesseract import image_to_string
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Rajkumar\AppData\Local\Tesseract-OCR\tesseract.exe"

########## image or scanned image to Text '''
def img_to_text(filename):
    # Read image with opencv
    img = cv2.imread(filename)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # blur a bit to improve ocr accuracy
#    img = cv2.blur(img, (1, 1))
    cv2.imwrite("thres.png", img)
    # Recognize text with tesseract for python
    read = Image.open("thres.png")
    text = image_to_string(read, lang='eng')
    return text

#filename = 'files/handwritten.jpg'
#text = img_to_text(filename)

########## PDF or scanned pdf to Text '''
import PyPDF2 
from pdf2image import convert_from_path 


def pdf_to_text(filename):
    pdf = open(filename,'rb')
    pdfReader = PyPDF2.PdfFileReader(pdf)
    num_pages = pdfReader.numPages
    count = 0
    text = ""
    #The while loop will read each page
    while count < num_pages:
        page = pdfReader.getPage(count)
        count +=1
        text += page.extractText()
    if len(text)<42:
        text = text
    else:
        fileurl = os.getcwd() + '\\'+filename
        pages = convert_from_path(fileurl)
        image_counter = 1
        text = ""
        for page in pages: 
            filename = "page_"+str(image_counter)+".jpg"
            page.save(filename, 'JPEG') 
            image_counter = image_counter + 1
        filelimit = image_counter-1
#        outfile = "out_text.txt"
#        f = open(outfile, "a")
        for i in range(1, filelimit + 1): 
            filename = "page_"+str(i)+".jpg"
            text += image_to_string(Image.open(filename))
            text = text.replace('-\n', '')      
#            f.write(text) 
#        f.close()
    return text
#filename = 'files/sample1.pdf'
#text = pdf_to_text(filename)

###### Docx & pptx to text '''
import textract
def other_to_text(filename):
    fileurl = os.getcwd() + '\\'+filename
    text = textract.process(fileurl)
    text = text.decode("utf-8") 
    text = text.replace('\n', '') 
    return text
 

#filename = 'files/dc.docx'
#filename = 'files/test.pptx'
#text = other_to_text(filename)

def extract_text(file):
    extension = file.split('.')[-1]
    if extension in ['jpg', 'png', 'jpeg']:
        text = img_to_text(file)
    elif extension == 'pdf':
        text = pdf_to_text(file)
    elif extension in ['docx', 'doc', 'pptx', 'ppt']:
        text = other_to_text(file)
    else:
        text = 'No Detected'
    return text
        
filename = 'files/axa.jpg'        
text = extract_text(filename)
      
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # check if there is a file in the request
        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')
        file = request.files['file']
        print(file)
        # if no file is selected
        if file.filename == '':
            return render_template('upload.html', msg='No file selected')
        extracted_text = extract_text(file)
        # extract the text and display it
        return render_template('upload.html',
                               msg='Successfully processed',
                               extracted_text=extracted_text)
    elif request.method == 'GET':
        return render_template('upload.html')

if __name__ == '__main__':
    app.run(port=8766)















