# fcn-mobilenet
The fork is intended for using Mobilenet 0.25 128 FCN on Orange PI computer vision system<BR>
mobilenet_v1_1.0_224 <BR>
#MODEL_URL = 'http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz'<BR>
replaced with mobilenet_v1_0.25_128  Which is 4 times faster <BR>
MODEL_URL = 'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128.tgz'<BR>
And everything that comes with it.

The project implments Fully Convolutional Network based on MobileNet v1  image_squeeze 0.25 image_size 128

To run the program you should have about a hundred annotated images stored in<BR>
Data_zoo/camvid/train<BR>
Data_zoo/camvid/trainannot<BR>
Data_zoo/camvid/test<BR>
Data_zoo/camvid/testannot<BR>
Data_zoo/camvid/val<BR>
Data_zoo/camvid/valannot<BR>
"Annotated image" means a pair of of PNG images with the same names. An image itself and it's annotation.
Images would sit in ".../train/" folder their annotations in "trainannot/" folder. Same for "test/" and "val/" folders.<BR>
Images supposed to be PNG RGB 24Bit color depth.
Label image is a graiscale PNG with 8bit color depth painted in #00 a background, color #01 for objects of class 1, color #02 for class 2 and so on up to NUM_OF_CLASSES which are 12 in this repo but yo may set it to other values up to 255.<br>
Labeling could be conveniently done with Pixel Annotation tool see https://github.com/abreheret/PixelAnnotationTool<BR>

The programm has 3 modes "train", "test", "visualise"<BR>

To train the network. In Fcn.py set variable mode = "train".<BR>
It took about 6 hours on a GTX1080 to converge the network on my dataset of 100 images.  
  
To validate the network. Set "mode" to "validate"  <BR>
It'll print out average losses for each class.
  
To see how the network will label images set "mode" to "visualize"  <BR>
It'll pick N images (N=batch size, to be exact) from Validation set and then create a few result images in a "logs/" folder.
