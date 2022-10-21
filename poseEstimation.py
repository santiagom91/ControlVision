##########################################################################################################
################################### Pose Estimation ######################################################
##########################################################################################################
#pip install mediapipe
import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.mediapipe.python.solutions.pose # for use mediapipe
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) # test with 0,1,2 depend where is located your webcam

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for  id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            lm.x*w, lm.y*h
            cx, cy = int(lm.x * w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 1, (0,0,255), cv2.FILLED) 
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, 'FPS: ' + str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    cv2.imshow('Image', img)

    cv2.waitKey(1)
