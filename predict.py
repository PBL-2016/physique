# coding: utf-8

from arg import Arguments
from rect import Rectangle
from utils import resize
import contour as C
from contour import recognize_contour
import cv2
import sys

import numpy as np

import chainer
from chainer import Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from model import PBLLogi

USAGE = """Usage: predict image-path

Output the most suitable fashion cluster.
"""

PRIMITIVE_BORDER = (0, 0, 255)
SKIN_BORDER = (255, 0, 0)
HAIR_BORDER = (0, 255, 0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_mcs_upperbody.xml')

model = None
optimizer = None

def init_model():
    global model, optimizer
    model = PBLLogi()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    serializers.load_hdf5('pbllogi.model', model)
    serializers.load_hdf5('pbllogi.state', optimizer)

def predict(sr, sg, sb, hr, hg, hb, ratio, contour):
    x = Variable(np.array([[sr/255, sg/255, sb/255, hr/255, hg/255, hb/255, \
        ratio, contour]]).astype(np.float32), volatile='on')
    y = model.fwd(x)
    cluster = np.argmax(y.data[0, :])
    return cluster

if __name__ == '__main__':
    arg = Arguments(sys.argv)
    fullpath = arg.get_as_file(0)

    # Check the file is exist
    if fullpath is None:
        sys.stderr.write(USAGE)
        exit(1)

    print('File:\t', fullpath)
    img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
    img = resize(img, 300)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    print('Face detect:\t', len(faces) > 0)
    bodies = body_cascade.detectMultiScale(gray, 1.1, 3)
    print('Body detect:\t', len(bodies) > 0)
    if len(faces) <= 0 or len(bodies) <= 0:
        exit(1)

    face = Rectangle(faces[0][0], faces[0][1], faces[0][2], faces[0][3])
    body = Rectangle(bodies[0][0], bodies[0][1], bodies[0][2], bodies[0][3])

    # Skin Section
    skinR = skinG = skinB = 0

    for col in range(face.half_w):
        for row in range(10):
            pixel = img[face.y+face.half_h-row+5, face.x+face.quarter_w+col]
            skinB += pixel[0]
            skinG += pixel[1]
            skinR += pixel[2]

    averageDivisor = face.half_w * 10
    skinR = int(skinR / averageDivisor)
    skinG = int(skinG / averageDivisor)
    skinB = int(skinB / averageDivisor)

    # Hair Section
    hairR = hairG = hairB = 0

    top = face.y - 5
    if top < 0:
        top = 0

    for col in range(face.half_w):
        for row in range(10):
            pixel = img[top+row, face.x+face.quarter_w+col]
            hairB += pixel[0]
            hairG += pixel[1]
            hairR += pixel[2]

    hairR = int(hairR / averageDivisor)
    hairG = int(hairG / averageDivisor)
    hairB = int(hairB / averageDivisor)

    print(("Skin Color:\033[38;2;{0};{1};{2}m■\033[0m" +
          "\tR={0}\tG={1}\tB={2}").format(skinR, skinG, skinB))
    print(("Hair Color:\033[38;2;{0};{1};{2}m■\033[0m" +
          "\tR={0}\tG={1}\tB={2}").format(hairR, hairG, hairB))

    # Body/Face Section
    body_ratio = body.w / face.w
    print('Body/Face:\t', body_ratio)

    # Contour Section
#    C.ready()
#    contour = recognize_contour(img[face.y:face.y2, face.x:face.x2])
#    print('Contour:\t', contour)

    # Prediction
    # Default contour type is 1(square)
    # 0(triangle) and 2(circle)
    init_model()
    print("Cluster:\t", predict(skinR, skinG, skinB, hairR, hairG, hairB, \
            body_ratio, 2))

#    print("Cluster:\t", predict(skinR, skinG, skinB, 18, 5, 2, 2.7, 1))

    # Show detect rectangles
    # Face border
    cv2.rectangle(img, (face.x, face.y),
                  (face.x2, face.y2), PRIMITIVE_BORDER, 2)
    # Body border
    cv2.rectangle(img, (body.x, body.y),
                  (body.x2, body.y2), PRIMITIVE_BORDER, 2)
    # Skin border
    cv2.rectangle(img, (face.x+face.quarter_w, face.y+face.quarter_h),
                  (face.x+face.quarter_w*3, face.y+face.half_h+5),
                  SKIN_BORDER, 2)
    # Hair border
    cv2.rectangle(img, (face.x+face.quarter_w, top),
                  (face.x+face.quarter_w*3, top+10),
                  HAIR_BORDER, 2)

    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    exit(0)
