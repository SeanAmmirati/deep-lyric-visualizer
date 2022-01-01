import os
import uuid
import eyed3

from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.mp3']
app.config['UPLOAD_MP3_FILEPATH'] = None
app.config['UPLOAD_LRC_FILEPATH'] = None


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    id_ = uuid.uuid4()

    if app.config['UPLOAD_MP3_FILEPATH']:
        os.remove(app.config['UPLOAD_MP3_FILEPATH'])
        os.remove(app.config['UPLOAD_LRC_FILEPATH'])

    app.config['UPLOAD_MP3_FILEPATH'] = os.path.join(
        'static/music', f'{id_}.mp3')
    app.config['UPLOAD_LRC_FILEPATH'] = os.path.join(
        'static/lyrics', f'{id_}.lrc')

    uploaded_mp3 = request.files['song']
    if uploaded_mp3.filename != '':
        uploaded_mp3.save(app.config['UPLOAD_MP3_FILEPATH'])
        if not eyed3.load(app.config['UPLOAD_MP3_FILEPATH']):
            os.remove(app.config['UPLOAD_MP3_FILEPATH'])
            return "Not valid file: Please try again"

    uploaded_lrc = request.files['lrc']
    if uploaded_lrc.filename != '':
        uploaded_lrc.save(app.config['UPLOAD_LRC_FILEPATH'])

    return redirect(url_for('index'))
