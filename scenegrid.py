import cv2
import math
'''
The script intended to explore transformations from screen coordinates to a desk surface coordinates.
It employs the usb camera to take an image of a page with printed square or a greed.
Then you have to (use a mouse) click 4 points in the corners of the square.
The script will calculate 2 wanishing points for the two sets of "parallel" lines.
Then it'll draw a coordinate grid over the square.
When mapping established it'll show mouse "coordinates" on the scene.
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
       

             
def drawLines4(img,tuplist):
    if len(tuplist)==4:       
       pointx=intersection(tuplist[0],tuplist[1],tuplist[2],tuplist[3])
       px1=pointx
       x,y=dif(pointx,tuplist[0])
       m1=math.atan2(y,x)
       x,y=dif(pointx,tuplist[3])
       m2=math.atan2(y,x)
       if m1>m2 : m1,m2=m2,m1
       dm =(m2-m1)/6
       m1-=2*dm
       while m1<=m2+2*dm:    
             a,b = farpoints(pointx,math.tan(m1))           
             cv2.line(img,a,b,(255,0,0),1)
             m1+=dm   
       pointx=intersection(tuplist[3],tuplist[0],tuplist[1],tuplist[2])
       px2=pointx
       x,y=dif(pointx,tuplist[0])
       m1=math.atan2(y,x)
       x,y=dif(pointx,tuplist[1])
       m2=math.atan2(y,x)
       if m1>m2 : m1,m2=m2,m1
       dm =(m2-m1)/6
       m1-=2*dm
       while m1<=m2+2*dm:    
             a,b = farpoints(pointx,math.tan(m1))           
             cv2.line(img,a,b,(255,0,0),1)
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
    if y==0:y=1
    if x==0:x=1
    return x,y
 
def slope(po1,po2):
    (x,y)=map(lambda a,b: a-b, po2, po1)
    if x==0 : return 10000
    else    : return y/x
      
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

def visible(x,y):
    return 0<=x<=640 and 0<=y<=480

def farpoints(po1,m1):
    x1,y1=po1
    points=[]
    x=0  ; y=m1*(x-x1)+y1  
    if visible(x,y) : points.append((round(x),round(y)))
    x=640; y=m1*(x-x1)+y1     
    if visible(x,y) : points.append((round(x),round(y)))
    y=0  ; x=(y-y1)/m1+x1
    if visible(x,y) : points.append((round(x),round(y)))
    y=480; x=(y-y1)/m1+x1
    if visible(x,y) : points.append((round(x),round(y)))
    
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
      drawLines1(img, tuplist) 
     # drawLines2(img,tuplist)
      # drawLines3(img,tuplist)
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
