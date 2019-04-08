# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:24:01 2019

@author: p
"""
from PIL import Image 
from os import listdir
from os.path import isfile, join

mypath="F:/TFP2/s2sLbl/33/"
destpath="F:/TFP2/s2sLbl/55/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for fi in onlyfiles :
    frame=Image.open(mypath+fi).convert("L")
    print( frame.mode)
    print( frame.getbands())
    frame.save(destpath+fi)