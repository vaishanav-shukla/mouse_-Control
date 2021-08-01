import cv2
import autopy
import time
import HandTrackingModulenew as htm
import mediapipe as mp
import numpy as np

##########
wcam,hcam=640,480
frameR=100
smoothening=8

# frame reduction (as we are not able to reach the
# bottom of the screen)
#########
pTime=0
plocx,plocy=0,0
clocx,clocy=0,0

cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)

detector=htm.handDetector(maxHands=1)
wScr,hScr=autopy.screen.size()
print(wScr,hScr)

while True:
    #1. landmark finding
    success, img = cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img)
    #2. get the tip of the index amd middle fingers
    if len(lmList)!=0:
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]
        # print(x1,x2,y2,y1)
        #3. check which fingers are up
        fingers=detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wcam - frameR, hcam - frameR)
                      , (255, 0, 254), 2)

    # 4. only index finger :moving mode
        if fingers[1]==1 and fingers[2]==0:
            # 5. convert coordinates

            x3=np.interp(x1,(frameR,wcam-frameR),(0,wScr))
            y3=np.interp(y1,(frameR,hcam-frameR),(0,hScr))

            #6. smoothen values
            clocx=plocx+(x3-plocx)/smoothening
            clocy=plocy+(y3-plocy)/smoothening

            #7. move mouse
            autopy.mouse.move(wScr-clocx,clocy)
            cv2.circle(img,(x1,y1),15,(225,0,225),cv2.FILLED)
            plocx,plocy=clocx,clocy
    #8. both index and midddle fingers are up :clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            length,img,lineInfo=detector.findDistance(8,12,img)
            print(length)
            # 9. find distance between fingera
            # 10. click mouse if discuss short
            if length<40:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),
                           15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()

    #11. frame rate
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,0),3)
    #12. display



    cv2.imshow("image",img)
    cv2.waitKey(1)