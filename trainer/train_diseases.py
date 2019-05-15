import action
import os
import argparse
import models.inception as a
import models.inception_resnet as b
import models.inception_resnet_se as c
import models.mobilenetV2 as d
import models.large_inception_resnet as e
import models.large_inception_resnet_se as f

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
      species_list = ['Apple','Badam','Blueberry','Cherry','Corn','Grape','Orange','Paddy','Peach','Pepper','Potato','Raspberry','Rose','Soyabean','Squash','Strawberry','Sunflower','Tomato']
      for species in species_list:
        #hyperParameters...
        action.event_dir='logs/{}/diseases/'+species+'/'+dataset+'_'+model_dict[model]+'event'
        action.base_dir='../../../datasets/Trainable/disease/'+dataset+'/'+species
        action.model_dir='../trainedModels/classifyDiseases/'+species+'/'+dataset+'_'+model_dict[model]+'.h5'
        action.train_history_dir='logs/diseases/'+species+'/'+dataset+'_'+model_dict[model]+'Hist'
        num_disease = len(next(os.walk(action.base_dir+'/'+'train'))[1])
        action.input_size=256; action.last_output=num_disease
        action.l1_out=64; action.l2_out=64
        action.l1_drop=0.5;action.l2_drop=0.5 
        action.batch_size=15
        action.epochs=250
        action.learn_rate=0.0001
      #classification only values for more than one class
        if num_disease>1:
          progress_num_local+=1
          if progress_num_local>progress_num_global:
            print("training "+species+" "+dataset+" for "+ model_dict[model])
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
      progress_file = "progress/progress_diseases"
      model_dict = {a:'inception',b:'inception_resnet'}
      models_list = [a, b]
      train(progress_file, model_dict, models_list)
    elif args.model == '2':
      progress_file = "progress/progress_diseases1"
      model_dict = {d:'mobilenetV2',c:'inception_resnet_se'}
      models_list = [d, c]
      train(progress_file, model_dict, models_list)
    elif args.model == '3':
      progress_file = "progress/progress_diseases2"
      model_dict = {e:'large_inception_resnet',f:'large_inception_resnet_se'}
      models_list = [e, f]
      train(progress_file, model_dict, models_list)
    else :
      print("Please select proper set of models")

    			
			