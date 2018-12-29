# To build, run "docker build -t dockerized_face_detector:face_morpher_service ."

from dockerized_face_detector:base

MAINTAINER tom.chau@oocl.com

ADD . /root/code
WORKDIR "/root/code"
CMD ["python", "face_morpher.py"]
