from PIL import Image
import numpy as np
from io import BytesIO
import cv2
import glob
import os, io
import base64

class FaceExtractor:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        return

    def from_base64_to_image(self, uri):
        """
        Read from Base64 image to OpenCV image
        Referred from: https://stackoverflow.com/questions/26070547/decoding-base64-from-post-to-use-in-pil
        """
        sbuf = BytesIO()
        sbuf.write(base64.b64decode(uri.split(",")[1]))
        pimg = Image.open(sbuf)
        return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    def from_image_to_base64(self, img):
        """
        Read from OpenCV image to Base64 image
        Referred from: https://stackoverflow.com/questions/40928205/python-opencv-image-to-byte-string-for-json-transfer
        """
        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)
        return str(jpg_as_text)

    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        """
        Resize a given image while maintaining ratio
        Referred from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
        """
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
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

    def detect_faces(self, src_base64_img):
        # Prepare return object
        return_dict = {
            'face_counts': 0,
            'faces': []
        }

        # Get greyscale image from base64 image
        img = self.from_base64_to_image(src_base64_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces with Haar Cascade
        #faces = face_cascade.detectMultiScale(gray, 1.2, 3, 0)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 10, 0)
        #faces = face_cascade.detectMultiScale(gray, 1.01, 20, 0)

        for (x,y,w,h) in faces:
            # Extract image for each detected face bounding box
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            face_img = img[y:y+h, x:x+w]

            resized_face_img = self.image_resize(face_img, height = 28)
            resized_face_img_base64 = self.from_image_to_base64(resized_face_img)
            # cv2.imshow('img',resized_face_img)
            # cv2.waitKey(0)

            return_dict['face_counts'] += 1
            return_dict['faces'].append(resized_face_img_base64)

        # cv2.destroyAllWindows()

        return return_dict

if __name__ == '__main__':
    # Unit test
    src_base64_img = ""         # Supply own image for testing

    fe = FaceExtractor()
    return_dict = fe.detect_faces(src_base64_img)
    print(return_dict['face_counts'])
    for img in return_dict['faces']:
        print(img[:100])
