# dockerized_face_detector
Dockerized example of a face detector site for 2018-Dec team sharing.

### Running the program
------

<b>1. Building the docker image per service:</b>

In project root, run:
<pre>docker build -t dockerized_face_detector:base . -f Dockerfile.base</pre>

- Place the Haar Cascade definition under face_extractor_service/haarcascades/haarcascade_frontalface_default.xml

- Place the pre-trained weights of DFC-VAE under face_morpher_service/checkpoint/dfc-vae3/

In face_extractor_service/, run:
<pre>docker build -t dockerized_face_detector:face_extractor_service .</pre>

In face_morpher_service/, run:
<pre>docker build -t dockerized_face_detector:face_morpher_service .</pre>

In web_server/, run:
<pre>docker build -t dockerized_face_detector:web_server .</pre>

<br />
<b>2. Running the services as docker containers:</b>

Make sure port 5000, 5051, 5052 are not used/listened by host machine:

face_extractor_service:
<pre>docker run -p 5051:5051 dockerized_face_detector:face_extractor_service</pre>

face_morpher_service:
<pre>docker run -p 5052:5052 dockerized_face_detector:face_morpher_service</pre>

web_server:
<pre>docker run -it -p 5000:5000 -e IS_DOCKER=1 -e FACE_EXTRACTOR_SERVICE_ENDPOINT=http://host.docker.internal:5051/extract -e FACE_MORPHER_SERVICE_ENDPOINT=http://host.docker.internal:5052/morph dockerized_face_detector:web_server bash</pre>

<br />
<b>3. Running the services with docker-compose:</b>

1. Make sure all docker images are built.
2. In project root, run:
<pre>docker-compose up</pre>
