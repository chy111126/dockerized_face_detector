from flask import Flask, flash, request, redirect, url_for, render_template
from flask import jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super secret key'
CORS(app)


@app.route('/extract', methods=['GET', 'POST'])
def extract_face_from_image():
    """
    Extract face using OpenCV + Haar Cascade detection
    :return:
    """
    req = request.form['image_data']
    print(req)
    return jsonify({'result': req})


app.run(debug=True, port=5051)
