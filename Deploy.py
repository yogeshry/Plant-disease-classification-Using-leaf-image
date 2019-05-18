from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import classifier

app = Flask(__name__)
CORS(app)

@app.route("/disease/<species_name>/<image_name>")
def get_disease(image_name,species_name):
    return jsonify(classifier.classify_disease('testImage/'+image_name,species_name))
@app.route("/species/<image_name>")
def get_species(image_name):
    return jsonify(classifier.classify_species('testImage/'+image_name))
		



if __name__ == '__main__':
     app.run(port='5000')
     