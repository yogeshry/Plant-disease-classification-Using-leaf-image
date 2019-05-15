from keras.models import load_model
import os
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

#load the segmentation_verification_model
model = load_model('../trainedModels/segmentation_verification/segmentation_verification_model.h5')

# Define our example directories and files
base_dir = '../../../datasets/Trainable/segmentation'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')



# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
	brightness_range=[0.5, 1.5],
	zca_whitening=True,
	channel_shift_range=0.1,
    horizontal_flip=True,
	vertical_flip=True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=10,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=10,
        class_mode='categorical')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)  
      
	 
model.save('../trainedModels/segmentation_verification/segmentation_verification_model.h5')	  