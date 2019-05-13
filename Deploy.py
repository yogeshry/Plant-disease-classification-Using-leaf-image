from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

import species_classifier
import disease_classifier

app = Flask(__name__)
CORS(app)

@app.route("/disease/<name>")
def get(self,name):
    return disease
@app.route("/species/<name>")
def get(self,name):
    return species
		



if __name__ == '__main__':
     app.run(port='5000')
     