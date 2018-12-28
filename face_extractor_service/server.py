from flask import Flask, flash, request, redirect, url_for, render_template

app = Flask(__name__)


@app.route('/extract', methods=['POST'])
def extract_face_from_image():
    """
    Extract face using OpenCV + Haar Cascade detection
    :return:
    """
    return


app.run()
