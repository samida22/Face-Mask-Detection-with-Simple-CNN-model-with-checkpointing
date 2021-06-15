#now we are going for detection parts
from keras.models import load_model
import cv2
import numpy as np


model = load_model('/home/kawaii/Desktop/FaceMaskDetections/model-008.model')

faceClassifier=cv2.CascadeClassifier('/home/kawaii/Desktop/FaceMaskDetections/haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

labels_dict={1:'MASK',0:'NO MASK'}
color_dict={0:(0,0,255),1:(0,255,0)}



while(True):

    ret,img=cap.read()
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceClassifier.detectMultiScale(gray_img,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=gray_img[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
cap.release()