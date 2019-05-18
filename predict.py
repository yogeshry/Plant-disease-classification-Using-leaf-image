import tensorflow as tf
from keras.preprocessing import image
import numpy as np
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

def predict(input_img_path,frozen_model_path="freeze/frozen_model.pb"):
  # load and prepare image for input as tensor 
  img = image.load_img(input_img_path, target_size=(256, 256))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = img/255
  print("check_load.................................")

  # We use our "load_graph" function
  graph = load_graph(frozen_model_path)

  # We can verify that we can access the list of operations in the graph
  #for op in graph.get_operations():
	  #print(op.name)
	  # prefix/Placeholder/inputs_placeholder  prefix/input_1
	  # ...
	  # prefix/Accuracy/predictions
 
  # We access the input and output nodes 
  x = graph.get_tensor_by_name(graph.get_operations()[0].name+':0')
  y = graph.get_tensor_by_name('prefix/k2tfout_0:0')
  print("check............................")
  # We launch a Session
  with tf.Session(graph=graph) as sess:
	  # Note: we don't need to initialize/restore anything
	  # There is no Variables in this graph, only hardcoded constants 
	  y_out = sess.run(y, feed_dict={x: img}) # img is input tensor
	  return y_out