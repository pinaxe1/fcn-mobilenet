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
       for poix in intersections(tuplist):
          mina=4  # pi
          maxa=-4 #-pi :)
          for point in tuplist: 
             x,y=dif(poix,point)
             m=math.atan2(y,x)
             if m<mina : mina=m
             if m>maxa : maxa=m
       dm =(maxa-mina)/5
       mina-=dm*2
       while mina<=maxa+dm*2:    
           for point1 in farpoints(poix,math.tan(mina)):
             cv2.line(img,point1,poix,(0,255,0),1)
           mina+=dm   
         
       #x,y=mapToSurface(xo,yo,px1,px2)
       #text=str(x)+" "+str(y)
       #cv2.putText(img, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 

def drawLines3(img,tuplist):
    if len(tuplist)==4:
       for poix in intersections(tuplist):
          for point in tuplist: 
             cv2.line(img,point,poix,(0,255,0),1)


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

         
def intersections(points):
    po1,po2,po3,po4=tuplist
    m1=slope(points[1],points[0])
    m2=slope(points[2],points[3])
    x1,y1=points[0]
    x2,y2=points[2]
    if m1==m2 : 
       m1=m1+0.01 
    xs1=(m1*x1-m2*x2+y2-y1)/(m1-m2)
    ys1=m1*(xs1-x1)+y1

    m1=slope(points[1],points[2])
    m2=slope(points[0],points[3])
    x1,y1=points[0]
    x2,y2=points[1]
    if m1==m2 : 
       m1=m1+0.01 
    xs2=(m1*x1-m2*x2+y2-y1)/(m1-m2)
    ys2=m1*(xs2-x1)+y1
    print((round(xs1),round(ys1)),(round(xs2),round(ys2)))
    return ((round(xs1),round(ys1)),(round(xs2),round(ys2)))


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
      drawLines1(img,tuplist) 
      drawLines2(img,tuplist)
      drawLines3(img,tuplist)
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