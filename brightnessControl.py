import numpy as np
import cv2
import time
import HandTrackingModule as htm
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
#briPer = 0
brightness = sbc.get_brightness()
Area = 0

while(True):
    #Capture frame by frame
    success, img = cap.read()

    img = detector.findHands(img, draw = False)
    lmList, bbox = detector.findPosition(img, draw = True)
    if len(lmList) != 0:

        #Filter Based on size
        Area = (bbox[2] - bbox[0])*(bbox[3] - bbox[1])//100
        #print(Area)

        if 250 < Area < 950:

            #print(lmList[4], lmList[8])
            length, img, lineInfo = detector.findDistance(4, 8, img)


            #cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            #print(length)

            #Hand Range from 30 to 180
            #Brightness Range from 0 to 100
            
            brightness = np.interp(length, [30, 180], [0, 100])
            briBar = np.interp(length, [30, 180], [400, 150])

            brightness = int(brightness)
            
            #print(int(length), brightness)
            smoothness = 10
            brightness = smoothness * round(brightness/smoothness)

            fingers = detector.fingersUp()

            if not fingers[3]:
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