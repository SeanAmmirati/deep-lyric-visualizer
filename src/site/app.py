import os
import uuid
import eyed3

import shutil
from flask import Flask, redirect, render_template, request, url_for, request
import time

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.mp3']
app.config['ID'] = None
app.config['UPLOAD_MP3_FILEPATH'] = None
app.config['UPLOAD_LRC_FILEDIR'] = None
app.config['UPLOAD_LRC_FILEPATH'] = None


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    id_ = uuid.uuid4()
    app.config['ID'] = id_
    remove_if_exists('UPLOAD_MP3_FILEPATH')

    app.config['UPLOAD_MP3_FILEPATH'] = os.path.join(
        'static/music', f'{id_}.mp3')

    uploaded_mp3 = request.files['file']
    if uploaded_mp3.filename != '':
        uploaded_mp3.save(app.config['UPLOAD_MP3_FILEPATH'])

    time.sleep(10)
    return redirect(url_for('step2'), code=302)
    # base_filename = os.path.splitext(uploaded_mp3.filename)[0]
    # os.system(f'cd ..; python deep_lyric_visualizer/visualize.py --song site/{app.config["UPLOAD_MP3_FILEPATH"]} --batch_size=5 --num_classes 3 --sort_classes_by_power 1 --subtitles 1 --truncation 1 --duration 2 --output_file {base_filename}.mp4')

@app.route('/step2' )
def step2():
    return render_template('step2.html')

@app.route('/step2', methods=["GET", "POST"])
def upload_lrc():
    remove_if_exists('UPLOAD_LRC_FILEDIR', True)
    id_ = app.config['ID']

    app.config['UPLOAD_LRC_FILEDIR'] = os.path.join(
        '../../data/lyrics/', str(id_))
    app.config['UPLOAD_LRC_FILEPATH'] = os.path.join(
        app.config['UPLOAD_LRC_FILEDIR'], f'{id_}.lrc'
    )

    os.mkdir(app.config['UPLOAD_LRC_FILEDIR'])
    print(app.config['UPLOAD_LRC_FILEDIR'])
    uploaded_lrc = request.files['file']
    if uploaded_lrc.filename != '':
        uploaded_lrc.save(app.config['UPLOAD_LRC_FILEPATH'])


@app.route('/config')
def config():
    return render_template('config.html')


def remove_if_exists(key, dir=False):
    if app.config[key]:
        if os.path.exists(app.config[key]):
            if dir:
                shutil.rmtree(app.config[key])
            else:
                os.remove(app.config[key])


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


    return redirect(url_for('index'))

@app.route('/processing')
def processing():
    return render_template('processing.html')

@app.route('/processing', methods=["POST"])
def processing_post():
    if request.method == 'POST':
        cmd = f'python deep_lyric_visualizer/visualize.py --song site/{app.config["UPLOAD_MP3_FILEPATH"]} --batch_size=5 --num_classes {request.form["numClasses"]}  --pitch_sensitivity={request.form["pitchSensitivity"]} --tempo_sensitivity={request.form["tempoSensitivity"]} --depth={request.form["depth"]} --jitter={request.form["jitter"]} --frame_length={request.form["frameLength"]} --smooth_factor={request.form["smoothFactor"]} --sort_classes_by_power 1 --subtitles {int("checked" == request.form["subtitles"])}  --truncation 1  --output_file' + (f' --duration {request.form["duration"]}' if request.form['duration'] else ' ') +  "render.mp4"
        os.system(f'cd ..;' +  cmd)


    return render_template('processing.html')
