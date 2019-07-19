import cv2
import numpy as np
i=0
k=0


cap=cv2.VideoCapture('FIFA World Cup 2018 France vs Croatia – Jul 15, 2018 - FullmatchTV.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 180000)
#cap=cv2.VideoCapture('Football in Ultra HD (2160p 4k).mp4')
#cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
if (cap.isOpened()== False):
    print("Error opening video file")

while(cap.isOpened()):
    i=i+1
    ret,image=cap.read()

    H, W, channels = image.shape 

    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    xxcolour1 = np.array([0,0, 0])
    xxcolour2 = np.array([255, 255, 255])
    maskxx = cv2.inRange(hsv, xxcolour1, xxcolour2) #blank image
    masknull = cv2.inRange(hsv, xxcolour1, xxcolour2)

    groundcolour1 = np.array([30,70, 70])
    groundcolour2 = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, groundcolour1, groundcolour2) #green range
    mask1 = cv2.bitwise_and(mask,masknull) #without crowd green peaple ?? 

    contours1,hierarchy1 = cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours1:
        x,y,w,h = cv2.boundingRect(c)
        if (h<=H/10 and w<=H/10):
            cv2.drawContours(mask1, [c], -1, 0, -1)

    mask2 = cv2.bitwise_not(mask1) #inverted
    mask3 = cv2.bitwise_and(mask2,masknull) #with big lines ??

    contours3,hierarchy3 = cv2.findContours(mask3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours3:
        cv2.drawContours(mask3, [c], -1, 255, int(H/50))

    mask3a = cv2.bitwise_and(mask3,masknull) #with big lines ??

    contours3a,hierarchy3a = cv2.findContours(mask3a,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours3a:    
        x,y,w,h = cv2.boundingRect(c)

        area = cv2.contourArea(c)
        rect_area = w*h
        extent = float(area)/rect_area
        
        if (h/w<=1.3 and w/h<=1.3 and h<=H/6 and w<=H/6 and extent>=0.65 and extent<=0.95):
            cv2.drawContours(maskxx, [c], -1, 0, -1)

    mask4 = cv2.bitwise_not(maskxx)         #best ball candidates
    mask5 = cv2.bitwise_and(mask2,mask4)    #remove thick lines
    mask6 = cv2.bitwise_and(mask5,masknull) #remove small dots ??

    contours5,hierarchy5 = cv2.findContours(mask6,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours5:    
        x,y,w,h = cv2.boundingRect(c)
        if (h<=H/100 or w<=H/100):
            cv2.drawContours(mask6, [c], -1, 0, -1)

    contours6,hierarchy6 = cv2.findContours(mask6,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours6:    
        x,y,w,h = cv2.boundingRect(c) 
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Ball', (x+w, y-10), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            
    image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #mask = cv2.resize(mask,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #mask1 = cv2.resize(mask1,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    mask2 = cv2.resize(mask2,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    mask3 = cv2.resize(mask3,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #mask4 = cv2.resize(mask4,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #mask5 = cv2.resize(mask5,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    mask6 = cv2.resize(mask6,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    
    cv2.imshow('image',image)
    #cv2.imshow('mask',mask)
    #cv2.imshow('mask1',mask1)
    cv2.imshow('mask2',mask2)
    cv2.imshow('mask3',mask3)
    #cv2.imshow('mask4',mask4)
    #cv2.imshow('mask5',mask5)
    cv2.imshow('mask6',mask6)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
