import action
import models.large_inception_resnet_se as a #edit
import CNN
import os

CNN.coreCNN = a.coreCNN #edit
dataset="original"  #edit
model="large_inception_resnet_se"  #edit
#hyperParameters...
action.event_dir='logs/{}/species/'+dataset+'_'+model+'event'
action.base_dir='../../../datasets/Trainable/species/'+dataset
action.model_dir='../trainedModels/classifySpecies/'+dataset+'_'+model+'.h5'
action.train_history_dir='logs/species/'+dataset+'_'+model+'Hist'
action.input_size=256; action.last_output=len(next(os.walk(action.base_dir+'/'+'train'))[1])
action.l1_out=256; action.l2_out=512
action.l1_drop=0.5;action.l2_drop=0.5 
action.batch_size=15
action.epochs=250
action.learn_rate=0.0001
print("training "+dataset+" for "+ model)
action.action.train()

