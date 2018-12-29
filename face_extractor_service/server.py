from flask import Flask, request, jsonify
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
    fe = FaceExtractor()

    input_json = request.get_json()
    src_base64_img = input_json['image_data']
    
    return_dict = fe.detect_faces(src_base64_img)

    return jsonify(return_dict)

app.run(debug=True, port=5051)
