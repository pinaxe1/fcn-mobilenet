# -*- coding: utf-8 -*-
"""
 The code derived from: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
 The script takes images from camera , performs semantic segmentation and commands robotic arm 
 to pick and throw away things classified as a trash.
  
 How it works in little more details:
 Acquires image from camera
 performs a semantic segmentation on the image
 checks random point on segmented image 
 if the point is classified as a trash transform its coordinates to robot arm angles and sends command to the arm over USB.
 To transform the point coordinates into arm angles transformation mesh should be prepared in advance with collectsamples.py script.
 It resides in Spatial-linear-interpolation-on-irregular-grid repository.
 
 The code loads pretrained model form file chk_pt="model.ckpt-1000"
 loads coordinate to arm angles mesh from file inf=open('datafile.dat', 'rb')
 
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import Utils as utils
from collections import namedtuple
import time
import cv2
import marshal
import serial  #
from scipy.interpolate import griddata

NUM_OF_CLASSES = 12
learning_rate =1e-4   #  "Learning rate for Adam Optimizer")
batch_size=1   # batch size for training")
logs_dir= "logs/"  # "path to logs directory")
data_dir= "Data_zoo/camvid/"  # "path to dataset")
model_dir = "Model_zoo/" # , "Path to mobile model mat")
chk_pt="model.ckpt-1000"
debug=False   #", "Debug mode: True/ False")
mode = "visualize"  # , "Mode train/ validate/ visualize")

IMAGE_NET_MEAN = [103.939, 116.779, 123.68]
IMAGE_SIZE = (480,640) # 320, 480)
MAX_ITERATION = int(1e5 + 1)
# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=8),
    DepthSepConv(kernel=[3, 3], stride=1, depth=16),
    DepthSepConv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=32),
    DepthSepConv(kernel=[3, 3], stride=2, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256)
]

def initSerial(ser):
    ser.port = "COM4"
    ser.baudrate = 9600
    ser.bytesize = serial.EIGHTBITS #number of bits per bytes
    ser.parity   = serial.PARITY_NONE #set parity check: no parity
    ser.stopbits = serial.STOPBITS_ONE #number of stop bits
    #ser.timeout = None          #block read
    ser.timeout  = 1            #non-block read
    #ser.timeout = 2              #timeout block read
    ser.xonxoff  = False     #disable software flow control
    ser.rtscts   = False     #disable hardware (RTS/CTS) flow control
    ser.dsrdtr   = False       #disable hardware (DSR/DTR) flow control
    ser.writeTimeout = 2     #timeout for write

    try: ser.open()
    except Exception as e: 
        print ("error open serial port: " + str(e))
        exit()

def sendCommand(ser,b,c,d) :
    if ser.isOpen():
        try:
            ser.flushInput() #flush input buffer, discarding all its contents
            ser.flushOutput()#flush output buffer, aborting current output 
                             #and discard all that is in buffer
            command="n100b10t100c10t"
            ser.write(command.encode('ASCII'))   # Send Arduino SnArm 3.1 command to flip output mode back OFF
            time.sleep(0.5)  #give the serial port sometime to receive the data
            command="n %dd %db %dc"%(d,b,c)                  
            print(command)
            ser.write(command.encode('ASCII'))   # Send Arduino SnArm 3.1 command to flip output mode back OFF
            time.sleep(0.5)  #give the serial port sometime to receive the data
        except Exception as e1:
            print ("error communicating...: " + str(e1))
    else:
        print ("cannot open serial port ")

def getbcd(xs,ys):
    global X,Y,B,C,D
    b=griddata((X, Y), B, (xs,ys), method='linear')
    c=griddata((X, Y), C, (xs,ys), method='linear')
    d=griddata((X, Y), D, (xs,ys), method='linear')
    return b,c,d

def loadData():
    inf = open('datafile.dat', 'rb')
    a = marshal.load(inf)
    b = marshal.load(inf)
    c = marshal.load(inf)
    d = marshal.load(inf)
    e = marshal.load(inf)
    inf.close(  )   
    return a,b,c,d,e

def MouseEventCallback(event, x, y, flags, param):
    global xs,ys,xo,yo,dataready
    if event == cv2.EVENT_LBUTTONUP:
        xo,yo=x,y
        dataready=True
    if event == cv2.EVENT_MOUSEMOVE:    
        xs,ys=x,y


def mobile_net(image, final_endpoint=None, num_classes=NUM_OF_CLASSES):
    with tf.variable_scope('MobilenetV1',reuse=tf.AUTO_REUSE):
        net = image
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
            for i, conv_def in enumerate(_CONV_DEFS):
                end_point_base = 'Conv2d_%d' % i
                if isinstance(conv_def, Conv):
                    net = slim.conv2d(net, conv_def.depth, conv_def.kernel,
                                      stride=conv_def.stride,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point_base)
                elif isinstance(conv_def, DepthSepConv):
                    end_point = end_point_base + '_depthwise'
                    net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                depth_multiplier=1,
                                                stride=conv_def.stride,
                                                rate=1,
                                                normalizer_fn=slim.batch_norm,
                                                scope=end_point)

                    end_point = end_point_base + '_pointwise'
                    net = slim.conv2d(net, conv_def.depth, [1, 1],
                                      stride=1,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)
                if final_endpoint and final_endpoint == end_point_base:
                    break
    return net


def inference(image, dropout_keep_prob, num_classes=NUM_OF_CLASSES):
    print("setting up mobile initialized conv layers ...")
    mean = tf.constant(IMAGE_NET_MEAN)
    image -= mean
    net = mobile_net(image, num_classes=num_classes)

    with tf.variable_scope('inference',reuse=tf.AUTO_REUSE):
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout')
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='Conv2d_1x1')
        net = slim.convolution2d_transpose(net, num_classes, 64, 32)

        annotation_pred = tf.argmax(net, axis=3, name="prediction")
        
    return tf.expand_dims(annotation_pred, dim=3), net


def main(argv=None):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('predi', cv2.WINDOW_NORMAL)

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder( tf.float32, shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], name="input_image")
    
    pred_annotation, logits = inference(image, dropout_keep_prob=keep_probability)
        
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, logs_dir+chk_pt)
    ###########################################
    global X,Y,B,C,D,dataready,xo,yo
    X,Y,B,C,D=loadData()
    ser = serial.Serial()
    initSerial(ser)
    cv2.setMouseCallback('predi', MouseEventCallback)
    ###########################################  
    
    while(cv2.getWindowProperty('predi', 0)>=0):
        _, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        rgb=utils.shape_image(frame,image_options)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
          break       
    
        images = rgb
        pred = sess.run(pred_annotation, feed_dict={image: images, keep_probability: 1.0})
 
        pred = np.squeeze(pred, axis=3)
        cv2.imshow('predi', utils.decode_segmap(pred[0]))
        #############
        if dataready: 
            b,c,d=getbcd(xo,yo) 
            sendCommand(ser,b,c,d)
            dataready=False
        #############        

    sess.close()
    cap.release()
    cv2.destroyAllWindows()

xs=ys=100
xo=yo=100
dataready=False
B=[]
C=[]
D=[]
X=[]
Y=[]
if __name__ == "__main__":
    tf.app.run()
