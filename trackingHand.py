###########################################################################################################
######################################## Tracking hand ####################################################
###########################################################################################################

# install: pip install mediapipe

import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.mediapipe.python.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) # test with 0,1,2 depend where is located your webcam

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
     
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id ==4:
                    cv2.circle(img, (cx, cy), 15, (255,0,0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, 'FPS: '+str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,2, (255,0,0),2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
