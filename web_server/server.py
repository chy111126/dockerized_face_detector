import os
from flask import Flask, request, redirect, url_for, render_template
from flask import jsonify
from flask_cors import CORS
import requests
from flask import Response
from flask import stream_with_context


app = Flask(__name__, template_folder='.')
app.config['SECRET_KEY'] = 'super secret key'
CORS(app)

if 'IS_DOCKER' in os.environ:
    # Prod deployment
    face_extractor_service_endpoint = os.environ['FACE_EXTRACTOR_SERVICE_ENDPOINT']     # e.g. 'http://127.0.0.1:5051/extract'
    face_morpher_service_endpoint = os.environ['FACE_MORPHER_SERVICE_ENDPOINT']         # e.g. 'http://127.0.0.1:5052/morph'
else:
    # Dev testing
    face_extractor_service_endpoint = 'http://127.0.0.1:5051/extract'
    face_morpher_service_endpoint = 'http://127.0.0.1:5052/morph'

@app.route('/extract', methods=['GET', 'POST'])
def extract_face_from_image():
    """
    Routes to internal service endpoint
    :return:
    """
    if request.method == 'POST':
        print(request.get_json())
        r = requests.post(face_extractor_service_endpoint, data=request.get_json())
        return Response(stream_with_context(r.iter_content()), content_type = r.headers['content-type'])


@app.route('/morph', methods=['GET', 'POST'])
def morph_faces():
    """
    Routes to internal service endpoint
    :return:
    """
    if request.method == 'POST':
        r = requests.post(face_morpher_service_endpoint, data=request.get_json())
        return Response(stream_with_context(r.iter_content()), content_type = r.headers['content-type'])


@app.route('/', methods=['GET'])
def index():
    """
    Index: returns default page
    :return:
    """
    return render_template('index.html')

app.run(host="0.0.0.0", port=5000)
