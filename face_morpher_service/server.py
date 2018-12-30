from flask import Flask, request, jsonify
from flask_cors import CORS
from face_morpher import FaceMorpher

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super secret key'
CORS(app)

fm = FaceMorpher()

@app.route('/morph', methods=['POST'])
def morph_faces():
    """
    Morph two faces using DFC-VAE + pre-trained model
    :return:
    """

    #input_json = request.get_json()
    input_json = request.form
    
    face1_src_base64_img = input_json['face1_data']
    face2_src_base64_img = input_json['face2_data']
    
    return_dict = fm.inference_model(face1_src_base64_img, face2_src_base64_img)

    return jsonify(return_dict)


app.run(host="0.0.0.0", port=5052)
