import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from django.shortcuts import render
import cv2

from main.src.yolo3.model import *
from main.src.yolo3.detect import *

from main.src.utils.image import *
from main.src.utils.datagen import *
from main.src.utils.fixes import *

from main.helpers import prepare_model, get_detection

import time as t

prepare_model(approach=2) 

fix_tf_gpu()
# from connector.cnxn import server_access
# from math import ceil


# Create your views here.
def index(request):
    return render(request, 'index.html')

def instructions(request):
    return render(request, 'instructions.html')

def awareness(request):
    return render(request, 'awareness.html')

def results(request):
    cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
    ret,img_cap = cap.read() # return a single frame in variable `frame`
    time_now = t.time()
    img_name = '{}.png'.format(time_now)
    # cv2.imwrite(r'main\static\images\{}'.format(img_name),img)
    cv2.destroyAllWindows()
    cap.release()

    # get the detection on the image
    # img = cv2.imread(r'main\static\images\1658401765.1840234.png')
    img, boxes = get_detection(img_cap)
    total = len(boxes[:, -1].tolist())
    non_complaint = boxes[:,-1].tolist().count(0.) + boxes[:,-1].tolist().count(1.)
    complaint = len(boxes[:,-1].tolist())
    print("Non Compliance: {}".format(non_complaint))
    # cv2.imwrite(r'main\static\images\result_{}.png'.format(time_now),img)
    cv2.imwrite(r'main\static\images\results\{}'.format(img_name),img_cap)
    if boxes[:,-1].tolist().count(0.) + boxes[:,-1].tolist().count(1.) > 0:  # not wearing PPE
        sign = 'cross.png'
        sound = 'wrong.mp3'
    else:
        sign = 'thumbs_up.gif'
        sound = 'applause.mp3'
    return render(request, 'results.html', context={'img_name':img_name, 'sign':sign, 'sound':sound,'total':total, 
    'complaint':complaint,'non_complaint':non_complaint})
