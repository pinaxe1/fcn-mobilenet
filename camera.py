# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:24:01 2019

@author: p
"""
import cv2
import datetime

cap = cv2.VideoCapture(0)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
while(True):
    _, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    cv2.imshow('frame', frame)
    wk = cv2.waitKey(1) & 0xFF
    if wk == ord('q'): cv2.imwrite(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.png', frame)
    if wk == ord('x'): break       
cap.release()
cv2.destroyAllWindows()       