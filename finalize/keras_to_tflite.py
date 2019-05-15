from tensorflow.contrib import lite

def convert(source,dest):
   converter = lite.TFLiteConverter.from_keras_model_file( source)
   tfmodel = converter.convert()
   open (dest , "wb") .write(tfmodel)