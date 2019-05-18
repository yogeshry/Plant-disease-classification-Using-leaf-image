import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

import matplotlib.pyplot as plt

pickle_in = open("../trainedModels/inferenceModels/inferenceModels_list","rb")
history = pickle.load(pickle_in)
model_name_list = []
for key in history.keys():
  model_name_list.append(history[key][0].split('_',1)[0])
# list all data in history
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.title.set_text("Overall summary of best classification models for all category")
ax1 = fig.add_subplot(322, projection=None)
ax1.axis('off')
ax2 = fig.add_subplot(341, projection=None)
ax2.axis('off')
ax3 = fig.add_subplot(324, projection=None)
ax3.axis('off')
i = 0
colors = ['r','g','b','c','m','y','k']
markers = ['^','o']
scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker=markers[-1])
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[1], marker=markers[-1])
scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[2], marker=markers[-1])
scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[3], marker=markers[-1])
scatter5_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[4], marker=markers[-1])
scatter6_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[5], marker=markers[-1])
scatter7_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker=markers[-1])
scatter8_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker=markers[0])
scatter9_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker=markers[-1])
scatter10_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$1$')
scatter11_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$2$')
scatter12_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$3$')
scatter13_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$4$')
scatter14_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$5$')
scatter15_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$6$')
scatter16_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$7$')
scatter17_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$8$')
scatter18_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$9$')
scatter19_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$10$')
scatter20_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$11$')
scatter21_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$12$')
scatter22_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$13$')
scatter23_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[6], marker='$14$')
models_name=[]
ax2.legend([scatter8_proxy,scatter9_proxy],  ['mixed', 'original'],   numpoints = 1, title = "Dataset Type")
ax1.legend([scatter1_proxy, scatter2_proxy,scatter3_proxy, scatter4_proxy,scatter5_proxy, scatter6_proxy,scatter7_proxy],
  ['inception', 'inception_resnet', 'inception_resnet_se','large_inception', 'large_inception_resnet', 'large_inception_resnet_se', 'mobilenetV2'], 
  numpoints = 1,title = 'Model Type')
ax3.legend([scatter10_proxy, scatter11_proxy,scatter12_proxy, scatter13_proxy,scatter14_proxy, scatter15_proxy,scatter16_proxy,scatter17_proxy, scatter18_proxy,scatter19_proxy, scatter20_proxy,scatter21_proxy, scatter22_proxy,scatter23_proxy],
  model_name_list, 
  numpoints = 1,title = 'Model Category')  

for key in history.keys():
  print(key)
  models_name = history[key][0]
  model=history[key][1]
  avg_score=(model[3]+model[4])/2
  implementation_score = model[0]
  i+=1
  x =[i]
  y =[avg_score]
  z =[implementation_score]
  if "mixed" in models_name:
    marker = markers[0]
  else:
    marker = markers[1]
  if "mobile" in models_name:
    color=colors[6]
  elif "large" in models_name:
    if "inception" in models_name:
      if "resnet" in models_name:
        if "se" in models_name:
          color=colors[5]
        else:
          color=colors[4]
      else:    
        color=colors[3]
  else:
    if "inception" in models_name:
      if "resnet" in models_name:
        if "se" in models_name:
          color=colors[2]
        else:
          color=colors[1]
      else:    
        color=colors[0]    
  
  ax.scatter(x, y, z,s = 70, c=color, marker=marker)
  ax.scatter([x[0]+1], y, z,s = 40, c=color, marker='$'+str(i)+'$')
     
ax.set_xlabel('model category')
ax.set_ylabel('average f1 score')
ax.set_zlabel('implementation_score')

plt.show()
