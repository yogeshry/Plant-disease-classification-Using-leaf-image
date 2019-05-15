import numpy as np
import pickle
import os
import keras_to_tf
import keras_to_tflite

source_dir1 = '../trainer/logs/diseases'
source_dir2 = '../trainer/logs/species'
model_name = 'a'
model_number = 0
models_dict = {}

#calculate model_score
def calculate_model_score(acc, val_acc, get_f1, val_get_f1, params_num):
  inference_score = (acc +val_acc+get_f1+val_get_f1)/4
  size_score = np.log(2000000/params_num)/np.log(20)
  percent_score = 100*(inference_score)+size_score
  return percent_score

# To empty the destination folder first
def empty_folder(folder):
    for the_file in os.listdir(folder):
      file_path = os.path.join(folder, the_file)
      try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
      except Exception as e:
        print(e)
#Choose, convert and save the best disease classification model for each species		
for a in os.listdir(source_dir1):
  model_score_max = 0
  #find the number of classes
  num_out = len(next(os.walk('../../../datasets/Trainable/disease/original/'+a+'/train'))[1])
  if num_out > 1:
    for b in os.listdir(source_dir1+'/'+a):
      pickle_in = open(source_dir1+'/'+a+'/'+b,"rb")
      example_dict = pickle.load(pickle_in)
      history = example_dict
      print(a+'/'+b)
      #calculate score of model looking training history
      model_score = calculate_model_score(max(history['acc']),max(history['val_acc']),max(history['get_f1']),max(history['val_get_f1']),history['params_num'])
      print(model_score)
	  #Update the best model
      if model_score>model_score_max:
        model_score_max = model_score
        print (b)
        model_name = b.replace("Hist",'')
	#save the best model score, name and number	
    models_dict[a] = [a+'_'+model_name+'.pb', model_score_max, model_number]
    model_number += 1
    folder_tf = '../trainedModels/inferenceModels/protobuf/diseases/'+a
    folder_tflite = '../trainedModels/inferenceModels/tflite/diseases/'+a
    empty_folder(folder_tf)
    empty_folder(folder_tflite)
    modelPath = '../trainedModels/classifyDiseases/'+a+'/'+model_name+'.h5'
    #convert and save to tf and tflite	
    keras_to_tf.keras_to_tf(modelPath,outdir=folder_tf,numoutputs=1,name =a+'_'+model_name+'.pb')
    keras_to_tflite.convert(modelPath,folder_tflite+'/'+a+'_'+model_name+'.tflite')
  
#Convert and save the best species classification model
model_score_max = 0
for a in os.listdir(source_dir2):
  pickle_in = open(source_dir1+'/'+a,"rb")
  example_dict = pickle.load(pickle_in)
  history = example_dict
  print(a)
  #calculate score of model looking training history
  model_score = calculate_model_score(max(history['acc']),max(history['val_acc']),max(history['get_f1']),max(history['val_get_f1']),history['params_num'])
  #Update the best model
  if model_score>model_score_max:
    model_score_max = model_score
    model_name = a.replace('Hist','')
#save the best model score, name and number		
models_dict['species'] = [model_name+'.pb', model_score_max,model_number]	
folder_tf = '../trainedModels/inferenceModels/protobuf/species'
folder_tflite = '../trainedModels/inferenceModels/protobuf/species'
empty_folder(folder_tf)
empty_folder(folder_tflite)
modelPath='../trainedModels/classifySpecies/'+model_name+'.h5'
#convert and save to tf and tflite	
keras_to_tf.keras_to_tf(modelPath,outdir=folder_tf,numoutputs=1,name =model_name+'.pb')
keras_to_tflite.convert(modelPath,folder_tflite+'/'+model_name+'.tflite')

print(models_dict)
#Save the result of conversion
with open('../trainedModels/inferenceModels/inferenceModels_list', 'wb') as file_pi:
       pickle.dump(models_dict, file_pi)