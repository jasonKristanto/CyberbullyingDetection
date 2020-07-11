import os

from flask import Flask
from flask import flash, jsonify, make_response, render_template, request, redirect, session, url_for

import detection

import pandas as pd
from gensim.models.fasttext import FastText

from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='views')
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect_text():
    if request.method == "POST":
        if 'text' in request.form and request.form['text'] != "":
            text = []
            text.insert(0, request.form['text'])
            
            predict_results = detection.cyberbullying_detection(text)

            result = dict()
            result['text'] = text[0]
            result['predict_result'] = predict_results[0]
            
            return render_template('index.html', result=result, category="text")
        else:
            flash('Input the text first to do cyberbullying detection', 'error_text')
            return redirect(url_for('index'))
    else:
        flash('Only POST method allowed', 'error_text')
        return redirect(url_for('index'))

@app.route('/detect_file', methods=['GET', 'POST'])   
def detect_file():
    if request.method == "POST":
        file = request.files['table']

        if 'table' in request.files and file and file.filename != "":
            table = pd.read_csv(file, header=None)

            predict_results = detection.cyberbullying_detection(table[0])

            results = dict()
            ctr_result = 0

            for predict_result in predict_results:
                if len(predict_results[predict_result]) != 0:
                    results[ctr_result] = dict()
                    results[ctr_result]['text'] = table[0][ctr_result]
                    results[ctr_result]['predict_result'] = predict_results[predict_result]
                    ctr_result += 1

            return render_template('index.html', results=results, category="table")
        else:
            flash('Upload the file first to do cyberbullying detection', 'error_file')
            return redirect(url_for('index'))
    else:
        flash('Only POST method allowed', 'error_file')
        return redirect(url_for('index')) 

app.run()