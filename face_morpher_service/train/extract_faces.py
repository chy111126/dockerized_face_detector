from PIL import Image
import cv2 as cv
import glob
import os


def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir('src_photo') if
             isfile(join('src_photo', f)) and f.endswith(".jpg")]

face_counts = 0

for fname in onlyfiles:
    img = cv.imread(join('src_photo', fname))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #faces = face_cascade.detectMultiScale(gray, 1.2, 3, 0)
    faces = face_cascade.detectMultiScale(gray, 1.3, 10, 0)
    #faces = face_cascade.detectMultiScale(gray, 1.01, 20, 0)

    for (x,y,w,h) in faces:
        #cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        roi_color = image_resize(roi_color, height = 64)

        cv.imwrite(join('extracted_faces', str(face_counts) + '.jpg'), roi_color)
        face_counts += 1
