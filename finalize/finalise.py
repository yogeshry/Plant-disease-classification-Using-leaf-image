import numpy as np
import pickle
import os
import keras_to_tf
source_dir1 = '../trainer/logs/diseases'
source_dir2 = '../trainer/logs/species'
model_name = 'a'
model_name_list = []
model_score_list = []

def empty_folder(folder):
    for the_file in os.listdir(folder):
      file_path = os.path.join(folder, the_file)
      try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
      except Exception as e:
        print(e)
for a in os.listdir(source_dir1):
  model_score_max = 0
  #num_out = len(next(os.walk('../../../datasets/Trainable/disease/original/'+a+'/train'))[1])
  if num_out > 1:
    for b in os.listdir(source_dir1+'/'+a):
      pickle_in = open(source_dir1+'/'+a+'/'+b,"rb")
      example_dict = pickle.load(pickle_in)
      history = example_dict
      print(a+'/'+b)
      #calculate score of model looking training history
      model_score = (max(history['acc'])+max(history['val_acc'])+max(history['get_f1'])+max(history['val_get_f1']))*25
      print(model_score)
      if model_score>model_score_max:
        model_score_max = model_score
        print (b)
        model_name = b.strip('Hist')
    model_name_list.append(a+'_'+model_name )
    model_score_list.append(model_score_max)
    folder = '../trainedModels/inferenceModels/protobuf/diseases/'+a
    empty_folder(folder)
    keras_to_tf.keras_to_tf(modelPath='../trainedModels/classifyDiseases/'+a+'/'+model_name+'.h5',outdir=folder,numoutputs=1,name =a+'_'+model_name+'.pb')
#   shutil.copyfile('../trainedModels/classifyDiseases/'+a+'/'+model_name+'.h5', '/path/to/other/phile') 
print(model_name_list) 
print(model_score_list)  

model_score_max = 0
#num_out = len(next(os.walk('../../../datasets/Trainable/species/original'+'/train'))[1])
for a in os.listdir(source_dir2):
  pickle_in = open(source_dir1+'/'+a,"rb")
  example_dict = pickle.load(pickle_in)
  history = example_dict
  print(a)
  #calculate score of model looking training history
  model_score = (max(history['acc'])+max(history['val_acc'])+max(history['get_f1'])+max(history['val_get_f1']))*25
  if model_score>model_score_max:
    model_score_max = model_score
    model_name = a.strip('Hist')
folder = '../trainedModels/inferenceModels/protobuf/species'
empty_folder(folder)
keras_to_tf.keras_to_tf(modelPath='../trainedModels/classifySpecies/'+model_name+'.h5',outdir=folder,numoutputs=1,name =model_name+'.pb')
#   shutil.copyfile('../trainedModels/classifyDiseases/'+a+'/'+model_name+'.h5', '/path/to/other/phile') 
print(model_name) 
print(model_score_max)  