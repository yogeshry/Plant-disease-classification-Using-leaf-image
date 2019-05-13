from keras.applications.nasnet import NASNetLarge
def coreCNN(img_input):        

  pre_trained_model = NASNetLarge(include_top=False, weights='imagenet', input_tensor=img_input)
  for layer in pre_trained_model.layers:
    layer.trainable = True
  return pre_trained_model.output
