from keras.applications.xception import Xception
def coreCNN(img_input):        

  pre_trained_model = Xception(include_top=False, weights='imagenet', input_tensor=img_input)
  for layer in pre_trained_model.layers:
    layer.trainable = True
  return pre_trained_model.output
