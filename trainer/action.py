import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend
from keras.preprocessing import image
from CNN import kerasModel
import os
import os.path as osp
from keras.models import load_model

class action():
  
  def init_model_class():

      return kerasModel(event_dir=event_dir,base_dir=base_dir, model_dir=model_dir, train_history=train_history_dir, input_size=input_size, last_output=last_output,
                          l1_out=l1_out, l2_out=l2_out, 
                          l1_drop=l1_drop, l2_drop=l2_drop, 
                          batch_size=batch_size, epochs=epochs,  
                          learn_rate=learn_rate)


     
  @classmethod
  def train(cls):
      _model = cls.init_model_class()
      print(_model.model().output)      #prints output_node_name
      cls.history = _model.evaluate()             #return evaluation

     
  @classmethod
  def show_history(cls):
      history = cls.history
      # list all data in history
      print(history.history.keys())
      # summarize history for accuracy
      plt.plot(history.history['acc'])
      plt.plot(history.history['val_acc'])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.show()
      # summarize history for loss
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.show()
      
  @classmethod    
  def prepare_graph_for_freezing(cls, model_folder="freeze/"):
      #model =  kerasModel(150,38,1024,512,0.5,0.2,20,5,0.00008)
      model = cls.init_model_class()
      checkpoint = tf.train.get_checkpoint_state(model_folder)
      input_checkpoint = checkpoint.model_checkpoint_path
      saver = tf.train.Saver()

      with tf.Session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
              backend.set_session(sess)
              saver.restore(sess, input_checkpoint)
              tf.gfile.MakeDirs(model_folder+'freeze')
              saver.save(sess, model_folder+'freeze/checkpoint', global_step = 0)
 
  @staticmethod             
  def freeze_graph(model_folder="freeze/"): 
      checkpoint = tf.train.get_checkpoint_state(model_folder)

      print(model_folder+'freeze/')
      input_checkpoint = checkpoint.model_checkpoint_path
      absolute_model_folder = '/'.join(input_checkpoint.split('/')[:-1])
      output_graph = absolute_model_folder + '/frozen_model.pb'

      output_node_names = 'y/Softmax'
      clear_devices = True
      new_saver = tf.train.import_meta_graph(input_checkpoint +'.meta',clear_devices = clear_devices)

      graph = tf.get_default_graph()
      input_graph_def = graph.as_graph_def()

      with tf.Session(config = tf.ConfigProto(allow_soft_placement = True)) as sess2:
        print(input_checkpoint)
        new_saver.restore(sess2,input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess2,
                                                                        input_graph_def,
                                                                        output_node_names.split(',')
                                                                        )
        with tf.gfile.GFile(output_graph,"wb") as f:
          f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        
  @classmethod
  def freeze_model(cls,model_folder="freeze/"):
      tf.reset_default_graph()
      cls.prepare_graph_for_freezing(model_folder)
      cls.freeze_graph(model_folder)   
	  
  

	  
  def load_graph(frozen_graph_filename):
      # We load the protobuf file from the disk and parse it to retrieve the 
      # unserialized graph_def
      with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())

      # Then, we import the graph_def into a new Graph and returns it 
      with tf.Graph().as_default() as graph:
          # The name var will prefix every op/nodes in your graph
          # Since we load everything in a new graph, this is not needed
          tf.import_graph_def(graph_def, name="prefix")
      return graph
   
  @classmethod
  def predict(cls, input_img_path,frozen_model_path="freeze/frozen_model.pb"):
      # load and prepare image for input as tensor 
      img = image.load_img(input_img_path, target_size=(256, 256))
      img = image.img_to_array(img)
      img = np.expand_dims(img, axis=0)
      img = img/255

      # We use our "load_graph" function
      graph = cls.load_graph(frozen_model_path)

      # We can verify that we can access the list of operations in the graph
      #for op in graph.get_operations():
          #print(op.name)
          # prefix/Placeholder/inputs_placeholder  prefix/input_1
          # ...
          # prefix/Accuracy/predictions

      # We access the input and output nodes 
      x = graph.get_tensor_by_name('prefix/input_1:0')
      y = graph.get_tensor_by_name('prefix/k2tfout_0:0')

      # We launch a Session
      with tf.Session(graph=graph) as sess:
          # Note: we don't need to initialize/restore anything
          # There is no Variables in this graph, only hardcoded constants 
          y_out = sess.run(y, feed_dict={x: img}) # img is input tensor
          return y_out
