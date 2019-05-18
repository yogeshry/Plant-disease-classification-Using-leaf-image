import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

import matplotlib.pyplot as plt

pickle_in = open("../trainedModels/inferenceModels/inferenceModels_list","rb")
history = pickle.load(pickle_in)
# list all data in history

for key in history.keys():
  print(key)
  if key!='species':
  
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.title.set_text("Summary for " +key+" diseases classification models")
    ax1 = fig.add_subplot(347, projection=None)
    ax1.axis('off')
    ax2 = fig.add_subplot(341, projection=None)
    ax2.axis('off')
    i = 0
    colors = ['r','g','b','c','m','y','r','g','b','c','m','y']
    marker = ['^','^','^','^','^','^','o','o','o','o','o','o']
    scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker=marker[-1])
    scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[1], marker=marker[-1])
    scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[2], marker=marker[-1])
    scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[3], marker=marker[-1])
    scatter5_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[4], marker=marker[-1])
    scatter6_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[5], marker=marker[-1])
    scatter7_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker=marker[0])
    scatter8_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker=marker[-1])
    ax2.legend([scatter7_proxy,scatter8_proxy],  ['mixed', 'original'],   numpoints = 1, title = "Dataset Type")
    ax1.legend([scatter1_proxy, scatter2_proxy,scatter3_proxy, scatter4_proxy,scatter5_proxy, scatter6_proxy],
    ['inception', 'inception_resnet', 'inception_resnet_se', 'large_inception_resnet', 'large_inception_resnet_se', 'mobilenetV2'], 
    numpoints = 1,title = 'Model Type')


    for model in history[key][3]:
      avg_acc=(model[1]+model[2])/2
      avg_f1score=(model[3]+model[4])/2
      params = model[5]
      x =[avg_acc]
      y =[avg_f1score]
      z =[params]
      ax.scatter(x, y, z,s = 60, c=colors[i], marker=marker[i])
      i+=1    
    ax.set_xlabel('average accuracy')
    ax.set_ylabel('average f1 score')
    ax.set_zlabel('size')

    plt.show()
  else:
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.title.set_text("Summary for " +key+" classification models")
    ax1 = fig.add_subplot(347, projection=None)
    ax1.axis('off')
    ax2 = fig.add_subplot(341, projection=None)
    ax2.axis('off')
    i = 0
    colors = ['r','g','b','c','r','g','b','c']
    marker = ['^','^','^','^','o','o','o','o']
    scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker=marker[-1])
    scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[1], marker=marker[-1])
    scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[2], marker=marker[-1])
    scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[3], marker=marker[-1])
   
    scatter7_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker=marker[0])
    scatter8_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker=marker[-1])
    ax2.legend([scatter7_proxy,scatter8_proxy],  ['mixed', 'original'],   numpoints = 1, title = "Dataset Type")
    ax1.legend([scatter1_proxy, scatter2_proxy,scatter3_proxy, scatter4_proxy],
    ['large_inception', 'large_inception_resnet', 'large_inception_resnet_se', 'mobilenetV2'], 
    numpoints = 1,title = 'Model Type')


    for model in history[key][3]:
      avg_acc=(model[1]+model[2])/2
      avg_f1score=(model[3]+model[4])/2
      params = model[5]
      x =[avg_acc]
      y =[avg_f1score]
      z =[params]
      ax.scatter(x, y, z,s = 60, c=colors[i], marker=marker[i])
      i+=1    
    ax.set_xlabel('average accuracy')
    ax.set_ylabel('average f1 score')
    ax.set_zlabel('size')

    plt.show()
