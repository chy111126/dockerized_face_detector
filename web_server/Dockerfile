# To build, run "docker build -t dockerized_face_detector:web_server ."

from dockerized_face_detector:base

MAINTAINER tom.chau@oocl.com

ADD . /root/code
WORKDIR "/root/code"
CMD ["python", "server.py"]
