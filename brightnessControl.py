import numpy as np
import cv2
import time
import HandTrackingModule as htm
import math
import screen_brightness_control as sbc

#####################################
wCam, hCam = 640, 480
#####################################

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0

detector = htm.handDetector(detectionConfidence = 0.7)

briBar = 400
briPer = 0
brightness = sbc.get_brightness()

while(True):
    #Capture frame by frame
    success, img = cap.read()

    img = detector.findHands(img, draw = False)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
        #print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cx, cy = (x1 + x2)//2, (y1 + y1)//2

        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        #cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        #print(length)

        #Hand Range from 30 to 180
        #Brightness Range from 0 to 100
        
        brightness = np.interp(length, [30, 180], [0, 100])
        briBar = np.interp(length, [30, 180], [400, 150])

        brightness = int(brightness)
        
        #print(int(length), brightness)

        sbc.set_brightness(brightness)

        #if length < 30:
           #cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0))
    cv2.rectangle(img, (50, int(briBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, "FPS: " + str(int(fps)), (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(img, "Brightness: " + str(int(brightness)) + "%", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    #Display the resulting frame
    cv2.imshow('Image', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#When everything done release the capture
cap.release()
cv2.destroyAllWindows()