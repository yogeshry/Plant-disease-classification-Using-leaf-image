## CNN-image-classification-for-plant-disease-identification
#### Major Project IOE pulchowk



### Requirements:
Look requirements.txt

### Dataset
Plant village dataset with few extra classes of images

link:  https://drive.google.com/open?id=1EuePtPjB2N_tZlaHjUTcw4p1JR4RldjP

### Preprocess the datasets
##### Segment the images (Otsu-segmentation and green pixel masking)
1. store the images in folder datasets/raw/species/disease-type
2. run  segment.py  inside preprocessing/image-segmentation
```
python segment.py
```


##### Verify segmentation and move well segmented images to augmented folder
``` 
python chk_move_segmented_image.py
```

**note: the segmentation_verification model is to be trained first which uses transfer learning in mobilenetV2 model.

```
python segmentation_verification_model_train.py
```

```
python resume_training.py
``` 
(to train for more epochs or updated data)

##### Move the images to Trainable folder in 80/20 ratio of train and validation set
```
python dataSplit.py --source original/augmented
```
### Train and test the various models
1. move to folder trainer
2. edit CNN models inside models
3. edit train_diseases.py and train_species.py for using the required models

4.
```
python train_diseases.py -m 1
```
 trains first group of models
5.
```
python train_diseases.py -m 2
``` 
 trains second group of models
###### Thus the training can be done in parallel
6. Similarly for species.

**note: The training is always resumed so use init_disease.py .... inside progress to restart training session**



### Visualize and compare models 
Use tensorboard to visualize events inside the logs/{}  Or use pickle files
```
tensorboard --logdir = "logs/{}"
```


### Finalize trainedModels
Choose best models with best modelScore = (max.acc+max.val_acc+max.f1_score+max.val_f1_score)/4 + log20(2000000/params)

convert to protobuf format remove unnecessary ops and save for inference
1. move to finalise 
2. 
```
python finalise.py
```

### Deploy using flask api
```
python deploy.py
```



#### References:

[Automatic Leaf Extraction from Outdoor
Images ](https://arxiv.org/pdf/1709.06437.pdf)

https://www.researchgate.net/publication/320649606_Plant_disease_identification_A_comparative_study

https://www.researchgate.net/publication/303336153_SVM_and_ANN_Based_Classification_of_Plant_Diseases_Using_Feature_Reduction_Technique

https://www.researchgate.net/publication/301879126_Advances_in_Very_Deep_Convolutional_Neural_Networks_for_LVCSR

#### Authors:

Krishna Upadhyay (krishnaupadhyay1997@gmail.com)

Sanjay Karki (sonJ9)

Simon Dahal (simonsd054@gmail.com)

Yogesh Rai (ygsh.spcry5@gmail.com)

