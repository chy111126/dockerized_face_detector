from flask import Flask, flash, request, redirect, url_for, render_template

app = Flask(__name__)


@app.route('/morph', methods=['POST'])
def morph_faces():
    """
    Morph two faces using DFC-VAE + pre-trained model
    :return:
    """
    return


app.run()
