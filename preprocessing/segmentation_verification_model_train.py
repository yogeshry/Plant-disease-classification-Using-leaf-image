import os
from keras import layers
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import Adam
	
#transfer learning with MobileNetV2	
img_input = layers.Input(shape=(150, 150,3))
pre_trained_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=img_input)
for layer in pre_trained_model.layers:
  layer.trainable = True
last_output = pre_trained_model.output 


# Flatten the output layer to 1 dimension
x = layers.GlobalAveragePooling2D()(last_output)
x= layers.BatchNormalization()(x)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(256, activation='elu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(2, activation='softmax')(x)

# Configure and compile the model
model = Model(pre_trained_model.input, x)
model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.00007),
                      metrics=['acc'])


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

# Flow validation images in batches of 10 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=10,
        class_mode='categorical')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=5,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)  
      
#save our segmentation_verification_model	 
model.save('../trainedModels/segmentation_verification/segmentation_verification_model.h5')	  