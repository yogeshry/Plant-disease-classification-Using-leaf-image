import os
import pickle
import predict
import numpy as np

Apple_diseases=['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy']
Badam_diseases=['bad','good']
Blueberry_diseases=['Blueberry___healthy']
Cherry_diseases=['Cherry_(including_sour)___healthy','Cherry_(including_sour)___Powdery_mildew']
Corn_diseases=['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___healthy','Corn_(maize)___Northern_Leaf_Blight']
Grape_diseases=['Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___healthy','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)']
Orange_diseases=['Orange___Haunglongbing_(Citrus_greening)']
Paddy_diseases=['Bacterial leaf blight','Brown spot','Leaf smut']
Peach_diseases=['Peach___Bacterial_spot','Peach___healthy']
Pepper_diseases=['Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy']
Potato_diseases=['Potato___Early_blight','Potato___healthy','Potato___Late_blight']
Raspberry_diseases=['Raspberry___healthy']
Rose_diseases=['anthracnose','mildew','rust','spot']
Soyabean_diseases=['Soybean___healthy']
Squash_diseases=['Squash___Powdery_mildew']
Strawberry_diseases=['Strawberry___healthy','Strawberry___Leaf_scorch']
Sunflower_diseases=['blight','mildew','rust']
Tomato_diseases=['Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus']

diseases = [Apple_diseases,Badam_diseases,Blueberry_diseases,Cherry_diseases,Corn_diseases,Grape_diseases,Orange_diseases,Paddy_diseases,Peach_diseases,Pepper_diseases,Potato_diseases,Raspberry_diseases,Rose_diseases,Soyabean_diseases,Squash_diseases,Strawberry_diseases,Sunflower_diseases,Tomato_diseases]
Species = ["Apple","Badam","Blueberry","Cherry","Corn","Grape","Orange","Paddy","Peach","Pepper","Potato","Raspberry","Rose","Soyabean","Squash","Strawberry","Sunflower","Tomato"]

model_detail = 'trainedModels/inferenceModels/inferenceModels_list'
def classify_species(imgPath):
  #open the list of inference models and build respective modelPath
  pickle_in = open(model_detail,"rb")
  models_dict = pickle.load(pickle_in)
  modelPath = "trainedModels/inferenceModels/protobuf/species/"+models_dict['species'][0]
  #the largest probability is provided for the class in prediction
  prediction = predict.predict(imgPath, modelPath)
  pred = np.argmax(prediction)
  pred_class = Species[pred]
  return [pred_class, str(max(prediction))]
  
def classify_disease(imgPath, species='unknown'):
  if species=='unknown':
     species = classify_species(imgPath)[0]
  #open the list of inference models and build respective modelPath
  pickle_in = open(model_detail,"rb")
  models_dict = pickle.load(pickle_in)
  modelPath = "trainedModels/inferenceModels/protobuf/diseases/"+species+'/'+models_dict[species][0]
  #the largest probability is provided for the class in prediction
  prediction = predict.predict(imgPath, modelPath)
  pred = np.argmax(prediction) 
  pred_class = diseases[models_dict[species][2]][pred]
  return [pred_class, str(max(prediction))]

