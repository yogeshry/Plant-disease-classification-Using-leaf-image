from keras.applications.inception_resnet_v2 import InceptionResNetV2
def coreCNN(img_input):        

  pre_trained_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=img_input)
  for layer in pre_trained_model.layers:
    layer.trainable = True
  return pre_trained_model.output