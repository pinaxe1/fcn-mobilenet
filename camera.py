import cv2
import math
from datetime import datetime

'''
The script intended to explore transformations from screen coordinates to a desk surface coordinates.
It employs the usb camera to take an image of a page with printed square or a greed.
Then you have to (use a mouse) click 4 points in the corners of the square.
The script will calculate 2 wanishing points for the two sets of "parallel" lines.
Then it'll draw a coordinate grid over the square.
'''
def drawLines1(img,tuplist):
    if len(tuplist)>1:         
       cv2.line(img,tuplist[0],tuplist[1],(0,0,255),4)
     
    if len(tuplist)>2:
       cv2.line(img,tuplist[1],tuplist[2],(0,0,255),4)

    if len(tuplist)>3:
       cv2.line(img,tuplist[2],tuplist[3],(0,0,255),4)
       cv2.line(img,tuplist[0],tuplist[3],(0,0,255),4)

    if len(tuplist)>4:
       del tuplist[:]
       

def drawLines2(img,tuplist):
    if len(tuplist)==4:
        
       pointx1=intersection(tuplist[0],tuplist[1],tuplist[2],tuplist[3])
       x,y=dif(pointx1,tuplist[0])
       m1=math.atan2(y,x)
       x,y=dif(pointx1,tuplist[3])
       m2=math.atan2(y,x)
       if m1>m2 : m1,m2=m2,m1
       dm =(m2-m1)/5
       m1-=dm
       while m1<=m2+dm:    
           for point1 in farpoints(pointx1,math.tan(m1)):
             cv2.line(img,point1,pointx1,(0,255,0),1)
           m1+=dm   
       px1=pointx1
       pointx1=intersection(tuplist[3],tuplist[0],tuplist[1],tuplist[2])
       px2=pointx1
       x,y=dif(pointx1,tuplist[0])
       m1=math.atan2(y,x)
       x,y=dif(pointx1,tuplist[1])
       m2=math.atan2(y,x)
       if m1>m2 : m1,m2=m2,m1
       dm =(m2-m1)/5
       m1-=dm
       while m1<=m2+dm:    
             for point1 in farpoints(pointx1,math.tan(m1)):
                cv2.line(img,point1,pointx1,(0,255,0),1)
             m1+=dm   
       x,y=mapToSurface(xo,yo,px1,px2)
       text=str(x)+" "+str(y)
       cv2.putText(img, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 

def mapToSurface(x,y,px1,px2):
    a,b=dif(px1,(x,y))
    m1=math.atan2(a,b)
    a,b=dif(px2,(x,y))
    m2=math.atan2(a,b)
    return m1,m2

def dif(po1,po2):
    x,y=map(lambda a,b: a-b, po2, po1)
    return x,y
 
def slope(po1,po2):
    (x,y)=map(lambda a,b: a-b, po2, po1)
    if x==0 : 
       return 1000
    else :
       return y/x

         
def intersection(po1,po2,po3,po4):
    m1=slope(po2,po1)
    m2=slope(po3,po4)
    x1,y1=po1
    x2,y2=po3
    if m1==m2 : 
       m1=m1+0.01 
    x=(m1*x1-m2*x2+y2-y1)/(m1-m2)
    y=m1*(x-x1)+y1
    return (round(x),round(y))


def farpoints(po1,m1):
    screenx=[0,640]
    screeny=[0,480]
    points=[]
    x1,y1=po1
    for x in screenx :
       points.append((x,round(y1+m1*(x-x1))))    
    for y in screeny :
       points.append((round(x1+(y-y1)/m1),y))    
    return points
   
def MouseEventCallback(event, x, y, flags, param):
    global tuplist,xo,yo
    
    if event == cv2.EVENT_LBUTTONUP:
        tuplist.append((x,y))
    if event == cv2.EVENT_MOUSEMOVE:    
        xo=x
        yo=y

def main(argv=None):
    global tuplist
    windowName = 'Drawing'
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, MouseEventCallback)
    while (True):
      _, img = cap.read()  
      # current date and time
      now = datetime.now()
      timestamp = datetime.timestamp(now)
      
      drawLines1(img, tuplist) 
      drawLines2(img,tuplist)
      cv2.imshow(windowName, img)
      key=cv2.waitKey(1) & 0xFF
      if key== ord('z'):
         filename=str(timestamp)+'.png' 
         cv2.imwrite(filename, img)
      if key== ord('c'):
         del tuplist[:]
      if key== ord('x'):
         break
    cv2.destroyAllWindows()

xo=0 
yo=0
if __name__ == "__main__":
   tuplist=[] 
   main()    