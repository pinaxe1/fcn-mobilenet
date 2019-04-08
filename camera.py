# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:34:17 2019
source: https://gist.github.com/cbednarski/8450931
Intended use:
To create and automatically annotate datasets. Pixelwize annotation.
1. Run Application.
2. Set camera steady with no moving or flickering objects on scene.
3. Precc "b" button. Camera will take background shot.
4. Put object to annotate on the scene. 
5. Press "q" button to take object shots.
6. The application wil take shot and create two png files. The object and the mask

@author: Pinaxe
"""
import cv2
import datetime

cap = cv2.VideoCapture(0)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
mask=[]
while(True):
    _, frame = cap.read()
    cv2.imshow('frame', frame)
    wk = cv2.waitKey(1) & 0xFF
    if wk == ord('b'):
        # cv2.imwrite(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-Back")+'.png', frame)
        backdrop=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if wk == ord('q'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     
        cv2.absdiff(backdrop,gray,mask)
        #mask = [0 if a_ > treshold else 255 for a_ in mask]
        mask[mask<4]=0
        cv2.imwrite(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-mask")+'.png', mask)
        #cv2.imwrite(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.png', gray)
    if wk == ord('x'):
        break      

cap.release()
cv2.destroyAllWindows()      
