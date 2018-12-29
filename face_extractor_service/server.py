from flask import Flask, flash, request, redirect, url_for, render_template
from flask import jsonify
from flask_cors import CORS
from face_extractor import FaceExtractor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super secret key'
CORS(app)


@app.route('/extract', methods=['GET', 'POST'])
def extract_face_from_image():
    """
    Extract face using OpenCV + Haar Cascade detection
    :return:
    """
    input_json = request.get_json()
    fe = FaceExtractor()
    src_base64_img = input_json['image_data']
    print(src_base64_img[:1000])
    return_dict = fe.detect_faces(src_base64_img)
    return jsonify(return_dict)


app.run(debug=True, port=5051)
