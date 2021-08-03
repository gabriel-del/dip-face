import numpy as np
import cv2
import matplotlib.pyplot as plt # Matplotlib

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = frame

    # Our operations on the frame come here
    face_cascade = cv2.CascadeClassifier('class/haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier('class/haarcascade_smile.xml')
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(cinza, scaleFactor=1.3, minNeighbors=5)
    len(faces)
    img_c = frame.copy()
    inpainted = frame.copy()
    mascara = frame.copy()
    mascara[:] = (0,0,0)
    for (x,y,w,h) in faces:
        # cv2.rectangle(img_c,(x,y),(x+w,y+h),(255,0,0),8)
        # rdi: região de interesse
        rdi = cinza[y:y+h, x:x+w]
        rdi_cor = img[y:y+h, x:x+w]
        sorriso = smile_cascade.detectMultiScale(rdi)
        # print(sorriso)
        for (ox,oy,ow,oh) in sorriso:
            # cv2.rectangle(img_c,(ox+x,oy+y),(ox+ow+x,oy+oh+y),(0,255,0),2)
            cv2.rectangle(mascara,(ox+x,oy+y),(ox+ow+x,oy+oh+y),(255,255,255),-1)
            mascara2 = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
            inpainted = cv2.inpaint(img_c, mascara2, 3, cv2.INPAINT_TELEA)

    # Display the resulting frame
    cv2.imshow('img',inpainted)
    # cv2.imshow('mascara',mascara2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
