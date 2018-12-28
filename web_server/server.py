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


@app.route('/extract', methods=['GET', 'POST'])
def extract_face_from_image():
    """
    Routes to internal service endpoint
    :return:
    """
    if request.method == 'POST':
        #print(request.get_json())
        #print(request.form['image_data'])
        r = requests.post('http://127.0.0.1:5051/extract', data=request.get_json())
        return Response(stream_with_context(r.iter_content()), content_type = r.headers['content-type'])


@app.route('/', methods=['GET'])
def index():
    """
    Index: returns default page
    :return:
    """
    return render_template('index.html')

app.run(debug=True)
