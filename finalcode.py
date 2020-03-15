#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################
import cv2
import argparse
import numpy as np
import numpy as np
import requests
import json
from statistics import mode 
from PIL import Image
import pytesseract
import datetime
import os

endTime = datetime.datetime.now() + datetime.timedelta(minutes=3/10)
  
list1 =[]

pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray) 
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 


def increase_size(image):
    scale_percent = 160 # percent of original size
    width = int(image.shape[1] * scale_percent / 100) 
    height = int(image.shape[0] * scale_percent / 100) 
    dim = (width, height) 
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
    return image

def blur(image):
     gray = cv2.medianBlur(image, 3)
     return gray

def adjust_light(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return image

def pyt(image):
    # show the output image
#print("[INFO] angle: {:.3f}".format(angle))
#cv2.imshow("Input", image)
#cv2.imshow("Rotated", rotated)
#cv2.waitKey(0)
                          
    #image = cv2.imread(imag)
    
    #image = get_grayscale(image)
    #image = remove_noise(image)
    #image = thresholding(image)
    #image = dilate(image)
    #image = erode(image)
    #image = opening(image)
    #image = canny(image)
    #image = deskew(image)
    #image = blur(image)
    #image = increase_size(image)
    #image = adjust_light(image)

    #image = match_template(image, template)
    
   

    custom_config = '--psm 6 --user-patterns C:/Program Files/Tesseract-OCR/tessdata/eng.user-patterns  -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    text = pytesseract.image_to_string(image, config=custom_config)
    text = text.replace(" ", "") 
    
    length = len(text)
    
    if length >1:
        if text[length-1]=='I':
            text  = text[:-1]
        
        list =['AP','AR','AS','BR','CG','GA','GJ','HR','HP','JH','KA','KL','MP','MH','MN','ML','OD','PB','RJ','SK','TN','TS','TR','UP','UK','WB','CH','DL','JK','LA','LD','PY']
        
      #  first_two = text[0:2]
        if len(text) >2:
            again_two = text[1:3]
            
            for i in range(0,len(list)):
                if again_two== list[i]:
                     text = text[1:]
                     break
            if text[2]=='O':
                new_text = text[:2] + '0' + text[3:]
                text = new_text
            first_two = text[0:2]
            
            count=0
            if len(text)-1>=0:
                index =len(text)-1
                while text[index]>='0' and text[index]<='9' and index>0:
                    count= count+1
                    index = index-1
                
                minus = count-4
                text = text[0:len(text)-minus]
                
                if count>=4:
                    for i in range(0,len(list)):
                        if first_two== list[i]:
                            print(text)
                            return text
                        
                

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img,lab, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, lab, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

k=0
vid=cv2.VideoCapture(0)



while(True):
  
    if datetime.datetime.now() >= endTime:
        break
    ret,image = vid.read()
    if ret==True:
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        
        classes = None
        with open("yolov3.txt", 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        net = cv2.dnn.readNet("yolo-obj_last.weights", "yolo-obj.cfg")
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        
        net.setInput(blob)
        
        outs = net.forward(get_output_layers(net))
        
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.7
        nms_threshold = 0.7
        
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.8:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = round(box[0])
            y = round(box[1])
            w = round(box[2])
            h = round(box[3])
            if confidences[i]> 0.8:
                roi=image[y:y+h,x:x+w]
                lab=pyt(roi)
                list1.append(lab)
                draw_prediction(image,lab, class_ids[i], confidences[i],x, y, x+w, y+h)
                cv2.imwrite("object-detection"+str(k)+".jpg", roi)
                k=k+1
        cv2.imshow("object detection", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()

#BELOW CODE SEND THE MOST FREQUENT NUMBER FROM THE LIST1 TO THE WEB SERVER WHERE DATABASE IS MAINTAINED FOR ENTRY OR EXIT OF VEHICLES
"""test1 =(mode(list1)) 
url = 'http://7ca86d11.ngrok.io/security/in'
body = {'vehicle': test1}
headers = {'content-type': 'application/json'}
requests.post(url, data=json.dumps(body), headers=headers)"""

