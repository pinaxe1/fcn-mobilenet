import cv2
import math
'''
The script intended to explore transformations from screen coordinates to a desk surface coordinates.
It employs the usb camera to take an image of a page with printed square or a greed.
Then you have to (use a mouse) click 4 points in the corners of the square.
The script will calculate 2 wanishing points for the two sets of "parallel" lines.
Then it'll draw a coordinate grid over the square.
'''
def drawLines1(img,tuplist):
    if len(tuplist)>1:         
       cv2.line(img,tuplist[0],tuplist[1],(0,255,0),1)
     
    if len(tuplist)>2:
       cv2.line(img,tuplist[1],tuplist[2],(0,255,0),1)

    if len(tuplist)>3:
       cv2.line(img,tuplist[2],tuplist[3],(0,255,0),1)
       cv2.line(img,tuplist[0],tuplist[3],(0,255,0),1)

    if len(tuplist)>4:
       del tuplist[:]
       

def drawLines3(img,tuplist):
    if len(tuplist)==4:
        
       pointx=intersection(tuplist[0],tuplist[1],tuplist[2],tuplist[3])
       m1=slope(pointx,tuplist[0])
       m2=slope(pointx,tuplist[3])
       if m1>m2 : m1,m2=m2,m1
       dm =(m2-m1)/6
       while m1<=m2:    
             cv2.line(img,farpoint (pointx,m1),pointx,(255,0,0),1)
             m1+=dm   
      
       pointx=intersection(tuplist[3],tuplist[0],tuplist[1],tuplist[2])
       m1=slope(pointx,tuplist[0])
       m2=slope(pointx,tuplist[1])
       if m1>m2 : m1,m2=m2,m1
       dm =(m2-m1)/6
       while m1<=m2:    
             cv2.line(img,farpoint (pointx,m1),pointx,(255,0,0),1)
             m1+=dm   
             
def drawLines4(img,tuplist):
    if len(tuplist)==4:
        
       pointx1=intersection(tuplist[0],tuplist[1],tuplist[2],tuplist[3])
       x,y=dif(pointx1,tuplist[0])
       m1=math.atan2(y,x)
       x,y=dif(pointx1,tuplist[3])
       m2=math.atan2(y,x)
       if m1>m2 : m1,m2=m2,m1
       dm =(m2-m1)/6
       while m1<=m2:    
             cv2.line(img,farpoint (pointx1,math.tan(m1)),pointx1,(0,255,0),1)
             m1+=dm   
      
       pointx1=intersection(tuplist[3],tuplist[0],tuplist[1],tuplist[2])
       x,y=dif(pointx1,tuplist[0])
       m1=math.atan2(y,x)
       x,y=dif(pointx1,tuplist[1])
       m2=math.atan2(y,x)
       if m1>m2 : m1,m2=m2,m1
       dm =(m2-m1)/6
       while m1<=m2:    
             cv2.line(img,farpoint (pointx1,math.tan(m1)),pointx1,(0,255,0),1)
             m1+=dm   


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

def farpoint(po1,m1):
    x1,y1=po1
    return (round(-x1),round(-2*m1*x1+y1))
   
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
      drawLines1(img, tuplist) 
     # drawLines2(img,tuplist)
      drawLines3(img,tuplist)
      drawLines4(img,tuplist)
      cv2.imshow(windowName, img)
      key=cv2.waitKey(1) & 0xFF
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
