from ner_predicter import NERPredicter
import __future__
import os
from flask import Flask, request, json, render_template, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

n = NERPredicter()

@app.route('/')
def hello_world():
	return 'Hello World'

@app.route('/ner_rio', methods = ['POST'])
def ner_rio():
	data_all = request.get_json()

	kalimat = data_all['kalimat']
	sent = kalimat
	out = ""
	for et in n.predict(sent):
		out += str(et) + "\n"
	return out

if __name__ == '__main__':
	app.run(host="0.0.0.0", port=5000)