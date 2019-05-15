from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import classifier

app = Flask(__name__)
CORS(app)

@app.route("/disease/<image_name>")
def get_disease(image_name):
    return classifier.classify_disease(image_name,'Apple')
@app.route("/species/<image_name>")
def get_species(image_name):
    return image_name
	#return classifier.classify_species(image_name)
		



if __name__ == '__main__':
     app.run(port='5000')
     