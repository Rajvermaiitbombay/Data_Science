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
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\SnehaD\AppData\Local\Tesseract-OCR\tesseract.exe"
import PyPDF2 
from pdf2image import convert_from_path
import textract 
from flask import Flask, render_template, request
from docx import Document
from pptx import Presentation

directory = os.getcwd()

''' image or scanned image to Text '''
def img_to_text(filename):
#   Read image with opencv
    img = cv2.imread(filename)
#   Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
#    blur a bit to improve ocr accuracy
#    img = cv2.blur(img, (1, 1))
    cv2.imwrite("thres.png", img)
    # Recognize text with tesseract for python
    read = Image.open("thres.png")
    text = image_to_string(read, lang='eng')
    return text

#filename = 'files/handwritten.jpg'
#text = img_to_text(filename)

''' PDF or scanned pdf to Text '''
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
    if len(text) != 0:
        text = text
    else:
        fileurl = directory + '\\'+filename
        pages = convert_from_path(fileurl, poppler_path=r'C:\Users\SnehaD\poppler-0.68.0\bin')
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

''' Docx & pptx to text '''
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
        
#filename = 'files/axa.jpg'        
#text = extract_text(filename)
      
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload_file.html')

@app.route('/upload')
def upload():
    if request.method == 'GET':
        return render_template('notify.html')
    return render_template('upload_file.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    global x
    if request.method == 'GET':
        return 'Sorry ! we did not get any dataset'
    elif request.method == 'POST' and request.files['myfile']:
        file = request.files['myfile']
        print(file)
        file = file
        extension = os.path.splitext(file.filename)[-1].lower()
        name = os.path.splitext(file.filename)[0].lower()
        if file.filename == '':
            return render_template('upload_file.html', msg= 'Please upload file!')
        file_path = directory + "/tmp//{0}{1}".format(name, extension)
        if extension in ['.png', '.jpeg', '.jpg']:
            try:
                file.save(file_path)
                text = img_to_text(file_path)
            except Exception:
                text = 'Not able to convert the text'
        elif extension in ['.doc', '.docx']:
            try:
                try:
                    document = Document(file)
                    document.save(file_path)            
                except PermissionError:
                    os.chmod(file_path, 0o777)
                    os.remove(file_path)
                    document = Document(file)
                    document.save(file_path)
                file_path = 'tmp//{0}{1}'.format(name, extension)
                text = other_to_text(file_path)
            except Exception:
                text = 'Not able to convert the text'
        elif extension in ['.pptx', '.ppt']:
            try:
                try:
                    document = Presentation(file)
                    document.save(file_path)            
                except PermissionError:
                    os.chmod(file_path, 0o777)
                    os.remove(file_path)
                    document = Presentation(file)
                    document.save(file_path)
                file_path = 'tmp//{0}{1}'.format(name, extension)
                text = other_to_text(file_path)
            except Exception:
                text = 'Not able to convert the text'
        elif extension == '.pdf':
            try:
                profile = PyPDF2.PdfFileReader(file)
                pages_no = profile.numPages
                output = PyPDF2.PdfFileWriter()
                for i in range(pages_no):
                    try:
                        profile = PyPDF2.PdfFileReader(file)
                        if profile.isEncrypted:
                            profile.decrypt("")
                        output.addPage(profile.getPage(i))
                    except Exception as e:
                        print(str(e))
        #        blob.writeBlob_stream(output, 'smple.pdf', container_name='rajkumar')
                with open("tmp/%s.pdf" % name, "wb") as out:
                    output.write(out)
                file_path = 'tmp/{0}{1}'.format(name, extension)
                text = pdf_to_text(file_path)
            except Exception:
                text = 'Not able to convert the text'
        else:
            text = 'Not able to convert the text'
        return text
#        return render_template('upload_file.html',
#                               msg='Successfully processed',
#                               extracted_text=text)

if __name__ == '__main__':
    app.run(port=6969)















