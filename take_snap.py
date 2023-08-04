

### Authors:        Rajdeep Chatterjee(TIFR), Shubham Dutta(SINP), Shilpi Jain (TIFR), Gregory Peter Powers (Boston University)
### acknowledgement: Shamik Ghosh (LLR)

import io
import time
from picamera2 import *
import cv2
import pytesseract
import numpy as np
from pytesseract import Output
from PIL import Image
import datetime
import json
from transform_funcs import find_rect_large, transform
from libcamera import controls

picam = Picamera2()

def takeSnapShot(filename):
    '''
    camera2 = Picamera2()
    #config =  camera2.still_configuration({"format": "NV12"})
    #camera2.configure(config)
    camera2.resolution = (640, 480)
    camera2.framerate = 30
    camera2.rotation = 180
    
    camera2.start()
    time.sleep(2)
    camera2.capture_file(f"{filename}")
    '''

    #picam = Picamera2()
    config = picam.create_still_configuration(transform=libcamera.Transform(hflip=1, vflip=1))
#    config = picam.create_still_configuration()
    picam.configure(config)

    # Set the resolution to a high value (e.g., 1920x1080)
    picam.start()
    picam.resolution = (2304, 1296)

    # Set the auto focus mode to continuous
    #picam.set_controls({"AfMode": controls.AfModeEnum.Continuous, "AfSpeed": controls.AfSpeedEnum.Fast})
    picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    time.sleep(2)
#    img = picam.capture_file(f"{filename}")
    img = picam.capture_array()
    if len(np.shape(img)) != 2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #contrast = 90
    #approx_num = 20

    contrast = 90
    approx_num = 20
    
    '''
    rect = find_rect_large(img,contrast,approx_num)
    ret, thresh = cv2.threshold(img, contrast, 255, 0)
    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(thresh,[rect],-1,(0,255,0),3)
    
    cv2.imshow('thresh',cv2.resize(thresh,(960,540)))
    cv2.resizeWindow('thresh',960,540)
    '''
    #processing
    warpstart = time.process_time()
    
    warped, rect = transform(img, contrast, approx_num)
#    ocr(warped)

    cv2.drawContours(img,[rect],-1,(0,255,0),3)
    #cv2.imshow("Image", img)
    #cv2.waitKey(0)
    cv2.imwrite(f"{filename.split('.')[0]}_original.png",img)
    cv2.imwrite(f"{filename}",warped)
#    cv2.destroyAllWindows()
    
    return warped
        


def isFloat(string):
    ### since sometimes decimals are represented by a comma (,), use this to convert it to decimal first
    string = string.replace(',', '.')
    try:
        float(string)
        return True
    except ValueError:
        return False




def cropped_image(image, x, y, w, h, alpha, beta):
    #alpha = 1.0
    #beta = -120.0
    #beta = -10.0
    if len(np.shape(image)) != 2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    img_gray = image
    #img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, 7)


    #kernel = np.ones((2,2),np.uint8)
    #img_gray = cv2.dilate(img_gray,kernel,iterations = 1)
    #img_gray = cv2.erode(img_gray,kernel,iterations = 1)
   
        
    

    # Define the coordinates of the ROI (crop box)
    # Crop the grayscale image to the defined ROI
    img_crop = img_gray[y:y+h, x:x+w]
    return img_crop


def read_image(x, y, w, h, results, images):

    coordinates = []
    for i in range(0, len(results["text"])):
        
        # We can then extract the bounding box coordinates
        # of the text region from  the current result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
      
        # We will also extract the OCR text itself along
        # with the confidence of the text localization

        #The OCR text itself along
        # with the confidence of the text localization
        text = results["text"][i]
        conf = int(results["conf"][i])
        # filter out weak confidence text localizations
        #if conf > args["min_conf"]:
        if conf > 0:
        
            # We will display the confidence and text to
            # our terminal
            print("Confidence: {}".format(conf))
            print("Text: {}".format(text))
            print("")
            coordinates.append(text)
            # We then strip out non-ASCII text so we can
            # draw the text on the image We will be using
            # OpenCV, then draw a bounding box around the
            # text along with the text itself
            text = "".join(text).strip()
            cv2.rectangle(images,
                          (x, y),
                          (x + w, y + h),
                          (0, 0, 255), 2)
#            cv2.putText(images,
#                        text,
#                        (x, y+20),
#                        cv2.FONT_HERSHEY_SIMPLEX,
#                        0.8, (0, 0, 255), 3)
            
        # After all, we will show the output image
        
    cv2.imshow("Image", images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return coordinates


def writeCoordinates(vert_pos_unit, hori_pos_unit, rot_axis_unit, filename):
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Separate date and time components
    current_date = current_datetime.date()
    current_time = current_datetime.time()

    # Format the date and time as a string to use as a suffix
    #suffix = current_datetime.strftime("%Y%m%d_%H%M%S")
    
    txt = "Date \t\t\t Time \t HorizontalPos \t\t VerticalPos \t RotationAxis\n"
#    txt += f"{suffix} \t {vert_pos_unit} \t {hori_pos_unit} \t {rot_axis_unit}"
    txt += f"{current_date} \t {current_time} \t {vert_pos_unit} \t {hori_pos_unit} \t {rot_axis_unit}"
     
    f = open(f"{filename}", "w")
    f.write(txt)
    f.close()

#### 





def processImage(image='', filename="", file_flag=False):

    if( file_flag ):
        img_source = cv2.imread(f'{filename}')
        #print("file_img_source = ", type(img_source))
        #print("file_img_source shape = ", img_source.shape)
    else:
        img_source = image
        #print("image_img_source = ", type(img_source))
        #print("image_img_source shape = ", img_source.shape)
    
    
    ###################### vertical position#########################
    #x, y, w, h = 20, 220, 400, 50 ### works for topmost
    #x, y, w, h = 160, 235, 220, 35 ### works for topmost
    x, y, w, h = 58, 386, 550, 80 ### works for topmost
    brgh = -200
    cont = 3.0
    images = cropped_image(img_source, x, y, w, h, cont, brgh)
    #cv2.imshow("Image", images)
    #cv2.waitKey(2000)
    results = pytesseract.image_to_data(images, output_type=Output.DICT, config='--psm 12 --oem 1 -c tessedit_char_whitelist="0123456789.+-cmd "', lang='eng')

    ### for debugging
    #coordinates = read_image(x, y, w, h, results, images)

    #print(results['text'])
    
    vert_pos = (list(filter(lambda text: isFloat(text), results['text'])))[0]
    unit = (list(filter(lambda text: text.islower() or text.isupper(), results['text'])))[0]
    
    vert_pos_unit = vert_pos+" "+unit

    ### for debugging
    #coordinates = read_image(x, y, w, h, results, images)
    
    #print(f"Vertical pos is {vert_pos} {unit}")
    ###################### vertical position########################3
    
    
    
    
    # ###################### horizontal position########################3
    #x, y, w, h = 160, 400, 220, 35 ### works for
    x, y, w, h = 63, 758, 550, 80 ### works for
    brgh = -370.0
    cont = 4.2
    images = cropped_image(img_source, x, y, w, h, cont, brgh)
    #cv2.imshow("Image", images)
    #cv2.waitKey(2000)
    results = pytesseract.image_to_data(images, output_type=Output.DICT, config='--psm 7 --oem 1 -c tessedit_char_whitelist="0123456789.+-cmd "', lang='eng')
    
    #print(results['text'])

    ### for debugging
    #coordinates = read_image(x, y, w, h, results, images)

    hori_pos = (list(filter(lambda text: isFloat(text), results['text'])))[0]
    unit = (list(filter(lambda text: text.islower() or text.isupper(), results['text'])))[0]

    hori_pos_unit = hori_pos+" "+unit
    
    #print(f"Horizontal pos is {hori_pos} {unit}")
    ######################## horizontal position#########################
    

    
    # ###################### Rotation Axis############################
    # #x, y, w, h = 160, 560, 250, 50 ### works for
    #x, y, w, h = 20, 1115, 620, 90 ### works for
    x, y, w, h = 50, 1122, 620, 80 ### works for
    brgh = -280.0
    cont = 2.9
    images = cropped_image(img_source, x, y, w, h, cont, brgh)
    #cv2.imshow("Image", images)
    #cv2.waitKey(2000)
    results = pytesseract.image_to_data(images, output_type=Output.DICT, config='--psm 11 --oem 1 -c tessedit_char_whitelist="0123456789.+-degrsain "', lang='eng')
    
    #print(results['text'])
    
    # ### for debugging
    #coordinates = read_image(x, y, w, h, results, images)
    
    rot_axis = (list(filter(lambda text: isFloat(text), results['text'])))[0]
    unit = (list(filter(lambda text: text.islower() or text.isupper(), results['text'])))[0]

    rot_axis_unit = rot_axis+" "+unit
    #print(f"Rotation axis is {rot_axis} {unit}")
    ###################### Rotation Axis##########################



    
    return vert_pos_unit, hori_pos_unit, rot_axis_unit




def getTableCoordinates():

    filename = "image_shdutta.png"
    image = takeSnapShot(filename)
    #vert_pos_unit, hori_pos_unit, rot_axis_unit = processImage(filename=filename, file_flag=True)
        
    vert_pos_unit, hori_pos_unit, rot_axis_unit = processImage(image=image)

    #processImage(filename)
    
    #### prepare a JSON format ##
    data = {
        "vertical position": f"{vert_pos_unit}",
        "horizontal position": f"{hori_pos_unit}",
        "Rotation axis": f"{rot_axis_unit}"
    }
    
    json_data = json.dumps(data)
    print(json_data)

    # #### write in txt file if needed
    #outFilename = "coordinates.txt"
    #writeCoordinates(vert_pos_unit, hori_pos_unit, rot_axis_unit, outFilename)

    return json_data
    
def run():
    count = 0
    maxcnt = 10
    while count < maxcnt:
        try:
            coords = getTableCoordinates()
            picam.stop()
            break
        except:
            print('retrying')
            count += 1
            picam.stop()
    if count == maxcnt:
        return None
    else:
        return coords

def main():
    for i in range(50): # for debugging put value > 1
        json_data = getTableCoordinates()
        
    

if __name__ == "__main__":
    main()    
