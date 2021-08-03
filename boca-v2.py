import numpy as np
import cv2
import matplotlib.pyplot as plt # Matplotlib

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('class/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('class/haarcascade_smile.xml')
green = False
blue = False
white = False
inpaint = True

while(True):
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_c = img.copy()
    inpainted = img.copy()
    mascara = img.copy()
    mascara[:] = (0,0,0)
    for (x,y,w,h) in face_cascade.detectMultiScale(cinza, scaleFactor=1.3, minNeighbors=5):
        if green: cv2.rectangle(img_c,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2) 
        if blue: cv2.rectangle(img_c,(int(x),int(y+(h/2))),(int(x+w),int(y+h)),(255,0,0),8)
        # rdi = cinza[int(y):int(y+(h/2)), int(x):int(x+w)] #blue
        rdi = cinza[y:y+h, x:x+w]
        for (ox,oy,ow,oh) in smile_cascade.detectMultiScale(rdi):
            if white: cv2.rectangle( img_c ,(ox+x,oy+y),(ox+ow+x,oy+oh+y),(255,255,255),-1)
            cv2.rectangle(  mascara,(ox+x,oy+y),(ox+ow+x,oy+oh+y),(255,255,255),-1)
            # cv2.rectangle(img_c,(ox+x,oy+y+int(h/2)),(ox+ow+x,oy+oh+y+int(h/2)),(255,255,255),-1)
            
            inpainted = cv2.inpaint(img_c, cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY) , 3, cv2.INPAINT_TELEA)
        			
    cv2.imshow('img', inpainted if inpaint else img_c)
    # cv2.imshow('img',img_c)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        green = False if green else True
    elif key == ord('b'):
        blue = False if blue else True
    elif key == ord('w'):
        white = False if white else True
    elif key == ord('i'):
        inpaint = False if inpaint else True

cap.release()
cv2.destroyAllWindows()
