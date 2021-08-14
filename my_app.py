import os
import threading 

# FLASK IMPORTS
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask import Flask, request, make_response, send_file, redirect, url_for
import requests

from training_data_generation import DATA_PATH, GENERATED_PATH, input_training_data, clean_data, new_training_data, progress_api, generated_data, send_request_no

app = Flask(__name__)
CORS(app)


def save_input_file(training_file):
    if training_file:
        filename = secure_filename(training_file.filename)
        training_file.save(os.path.join(DATA_PATH(), filename))
    return filename


@app.route('/input_data', methods = ['POST'])
def input_data():
    if request.method == 'POST':
        filename = save_input_file(request.files['training_file'])
        #sheet_name = request.form['sheet_name']
        file_path = input_training_data(filename)
    return file_path

#Clean Data
@app.route('/cleaning_data', methods = ['POST'])
def cleaning_data():
    if request.method == 'POST':
        #filename = save_input_file(request.files['training_file'])
        #sheet_name = request.form['sheet_name']
        #file_path = clean_data(filename)
        file_path = clean_data()
    #return send_file(file_path, attachment_filename = 'cleaned_file.csv')
    return file_path
'''
@app.route('/generate_data', methods = ['POST'])
def generate_data():
    if request.method == 'POST':
        file_path = new_training_data()
    #return send_file(file_path, attachment_filename = 'generated_data.csv')
    return file_path
'''

@app.route('/generate_data', methods = ['POST'])
def generate_data_test():
    if request.method == 'POST':
        request_no = request.form['request_no']
        send_request_no(request_no)
        threading.Thread(target=new_training_data).start()
        #new_training_data()
    return 'data generation started'


@app.route('/get_pro', methods = ['POST'])
def get_pro_test():
    if request.method == 'POST':
        request_no = request.form['request_no']
        progress = progress_api(request_no)
    return progress

@app.route('/get_data', methods = ['POST'])
def get_data():
    if request.method == 'POST':
        request_no = request.form['request_no']
        data = generated_data(request_no)
    return data


if __name__=='__main__':
    app.debug = True
    app.run(port = 2000, host = '0.0.0.0')
