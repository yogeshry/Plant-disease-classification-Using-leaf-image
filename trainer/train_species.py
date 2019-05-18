import action
import os
import argparse
import models.large_inception as a
import models.large_inception_resnet as b
import models.large_inception_resnet_se as c

import models.mobilenetV2 as d
import CNN
import pickle

def train(progress_file, model_dict, models_list):
  pickle_in = open(progress_file,"rb")
  progress_num_global = pickle.load(pickle_in)
  progress_num_local = 0

  for model in models_list:
    CNN.coreCNN = model.coreCNN
    dataset_list = ["original", "mixed"]
    for dataset in dataset_list:
      #hyperParameters...
      action.event_dir='logs/{}/species/'+dataset+'_'+model_dict[model]+'event'
      action.base_dir='../../../datasets/Trainable/species/'+dataset
      action.model_dir='../trainedModels/classifySpecies/'+dataset+'_'+model_dict[model]+'.h5'
      action.train_history_dir='logs/species/'+dataset+'_'+model_dict[model]+'Hist'
      action.input_size=256; action.last_output=len(next(os.walk(action.base_dir+'/'+'train'))[1])
      action.l1_out=256; action.l2_out=512
      action.l1_drop=0.5;action.l2_drop=0.5 
      action.batch_size=15
      action.epochs=250
      action.learn_rate=0.0001
      progress_num_local+=1
      if progress_num_local>progress_num_global:
        print("training "+dataset+" for "+ model_dict[model])
        action.action.train()
        progress_num_global+=1
        with open(progress_file, 'wb') as file_pi:
          pickle.dump(progress_num_global, file_pi)
if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--model','-m', dest='model', required=True)
  args = parser.parse_args()
  #models selection
  if args.model == '1':
    progress_file = "progress/progress_species"
    model_dict = {a:'large_inception',b:'large_inception_resnet',c:'large_inception_resnet_se',d:'mobilenetV2'}
    models_list = [a, b, c, d]
    train(progress_file, model_dict, models_list)
  elif args.model == '2':
    progress_file = "progress/progress_species1"
    model_dict = {}
    models_list = []
    train(progress_file, model_dict, models_list)
         
              
       
    
  else :
    print("Please select proper set of models")