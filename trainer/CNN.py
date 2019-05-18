# coding: utf-8



# * Hyperparameters of the model include the followings:
# * - output shape of the first layer
# * - dropout rate of the first layer
# * - output shape of the second layer
# * - dropout rate of the second layer
# * - batch size
# * - number of epochs

# #### Import libraries 
import os
import tensorflow as tf
import pickle
import time
from keras import layers
from keras import Model
from keras import backend
from keras import callbacks
#from keras.metrics import categorical_crossentropy
#from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator




# #### Define MNIST model
# * includes data loading function, training function, fit function and evaluation function 

# In[2]:

class TFCheckpointCallback(callbacks.Callback):
      def __init__(self,saver,sess):
        self.saver = saver
        self.sess = sess
      #this callback will be called and save ckpt in every epoch
      def on_epoch_end(self,epoch, logs=None):
        self.saver.save(self.sess, 'freeze/checkpoint', global_step = epoch)
         
        
# kerasModel class
class kerasModel():
  #lets setup keras callbacks
   
  
    def __init__(self,event_dir = '', base_dir='/data', model_dir='/content/model.h5',train_history='/content/history', input_size=150, last_output=38,
                 l1_out=512, 
                 l2_out=512, 
                 l1_drop=0.2, 
                 l2_drop=0.2, 
                 batch_size=100, 
                 epochs=10, learn_rate=0.00001 
                 ):
      
        #using keras callback function to save trained parameters as ckpt
        tf.reset_default_graph()
        #create tensorflow session
        self.sess = tf.Session()
        #instruct keras to use the sess we created
        backend.set_session(self.sess)
        
        
	     	# img_input = layers.Input(shape=(self.__input_size, self.__input_size, 3))		 
        self.__input_size = input_size
        self.__last_output = last_output
        self.l1_out = l1_out
        self.l2_out = l2_out
        self.l1_drop = l1_drop
        self.l2_drop = l2_drop
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.event_dir = event_dir
        self.base_dir = base_dir
        self.model_dir = model_dir
        self.train_history = train_history
        self.__model = self.model()
        self.train_generator,self.validation_generator = self.data()
        #we instantiate the tf saver and use it to setup the callback
        self.tf_saver = tf.train.Saver(max_to_keep=2)
        self.checkpoint_callback = TFCheckpointCallback(self.tf_saver,self.sess)
        
    # load mnist data from keras dataset
    def data(self):
	
        train_dir = os.path.join(self.base_dir, 'train')
        validation_dir = os.path.join(self.base_dir, 'validation')
        # Adding rescale, rotation_range, width_shift_range, height_shift_range,
        # shear_range, zoom_range, and horizontal flip to our ImageDataGenerator
        train_datagen = ImageDataGenerator( rescale=1./255,
                                            rotation_range=40,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
											brightness_range=[0.5, 1.5],
                                            channel_shift_range=0.1,
	                                        vertical_flip=True)

		    # Note that the validation data should not be augmented!
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Flow training images in batches of 32 using train_datagen generator
        train_generator = train_datagen.flow_from_directory(train_dir,  # This is the source directory for training images
                                                            target_size=(self.__input_size, self.__input_size),  # All images will be resized to self.__input_sizexself.__input_size
                                                            batch_size=self.batch_size,
                                                            # Since we use categorical_crossentropy loss, we need categorical labels
                                                            class_mode='categorical')

		    # Flow validation images in batches of 32 using test_datagen generator
        validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                                target_size=(self.__input_size, self.__input_size),
                                                                batch_size=self.batch_size,
                                                                class_mode='categorical')
		
        return  train_generator, validation_generator
    
    # keras model
    def model(self):
        


        img_input = layers.Input(shape=(self.__input_size,self.__input_size,3))

        # Flatten the output layer to 1 dimension
        x = layers.GlobalAveragePooling2D()(coreCNN(img_input))


        x = layers.BatchNormalization()(x)
        x = layers.Dense(self.l1_out, activation='relu')(x)
        # Add a dropout rate of l1_drop
        x = layers.Dropout(self.l1_drop)(x)
        
       

        # Add a final sigmoid layer for classification
        x = layers.Dense(self.__last_output, activation='softmax',name='y')(x)

        # Configure and compile the model
        model = Model(img_input, x)

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learn_rate),
                      metrics=['acc',self.get_f1])

        return model
    def get_f1(self,y_true, y_pred): #taken from old keras source code
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
        predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + backend.epsilon())
        recall = true_positives / (possible_positives + backend.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+backend.epsilon())
        return f1_val
    
    # fit mnist model
    def fit(self): 
        tensorboard = callbacks.TensorBoard(log_dir=self.event_dir,write_graph=True, write_images=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                                                patience=7, min_lr=0.000001,verbose=1)
        earlyStop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='auto', baseline=None, restore_best_weights=True)       
        self.history = self.__model.fit_generator(self.train_generator,
                                                  steps_per_epoch=100,
                                                  epochs=self.epochs,
                                                  validation_data=self.validation_generator,
                                                  validation_steps=50,
                                                  verbose=2,
                                                  callbacks=[tensorboard,reduce_lr,earlyStop])
        #save the train history as bytestream using pickle
        #print(self.__model.count_params())
        self.__model.save(self.model_dir)
        self.history.history['params_num'] = self.__model.count_params()
        with open(self.train_history, 'wb') as file_pi:
          pickle.dump(self.history.history, file_pi)
        

    
    # evaluate mnist model
    def evaluate(self):
        self.fit()
        return self.history
        #self.__model.save('my_model.h5')
        #evaluation = self.__model.evaluate_generator(self.validation_generator,50 )
        #return evaluation





