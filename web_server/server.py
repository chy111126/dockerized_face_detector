import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import uuid

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, template_folder='.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_image_folder():
    """
    Creates folders for face-extracted image
    :return:
    """
    img_uid = str(uuid.uuid4().hex)
    img_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], img_uid)
    face_output_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], img_uid, "output")

    if not os.path.exists(img_folder_path):
        os.mkdir(img_folder_path)

    if not os.path.exists(face_output_folder_path):
        os.mkdir(face_output_folder_path)

    return img_uid, img_folder_path, face_output_folder_path


@app.route('/extract', methods=['POST'])
def extract_face_from_image():
    """
    From
    :return:
    """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower()
            img_uid, img_folder_path, face_output_folder_path = create_image_folder()

            file.save(os.path.join(img_folder_path, 'image.' + file_ext))
            return redirect(url_for('upload_file',
                                    filename=img_uid))


@app.route('/', methods=['GET'])
def index():
    """
    Index: returns default page
    :return:
    """
    return render_template('index.html')

app.run()