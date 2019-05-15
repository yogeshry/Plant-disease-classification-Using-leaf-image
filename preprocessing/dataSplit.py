###Script to make datset

import os
import shutil
import random   
import argparse
  
# Update datasets

def update(source):
    
    source_dir = "../../../datasets/All"+"/"+source
    dest_dir1 = "../../../datasets/Trainable/disease/"+source+"/"
    dest_dir2 = "../../../datasets/Trainable/species/"+source+"/"
    dest_dir3 = "../../../datasets/Trainable/disease/"+"mixed"+"/"
    dest_dir4 = "../../../datasets/Trainable/species/"+"mixed"+"/"
    #open source directory
    for a in os.listdir(source_dir):
        for b in os.listdir(source_dir+"/"+a):
                #open images
                files = os.listdir(source_dir+"/"+a+"/"+b)
                total_files = len(files)
                if total_files == 0 :
                    print("No file to move from "+source_dir+"/"+a+"/"+b +" !")
                    
                else:
                    #initialize train images number list
                    train_image_nums = []
                    for eachNum in range(total_files):
                        train_image_nums.append(eachNum)
                    #20% images randomly chosen for validation    
                    test_image_nums = random.sample(range(total_files),int(total_files/5))
                    #move test images to targeted folder
                    for n in test_image_nums:
                        train_image_nums.remove(n)
                        shutil.copy(source_dir+"/"+a+"/"+b+"/"+files[n], dest_dir1+a+'/'+"validation"+'/'+b+'/'+files[n])
                        shutil.copy(source_dir+"/"+a+"/"+b+"/"+files[n], dest_dir2+"validation"+'/'+a+'/'+files[n])
                        shutil.copy(source_dir+"/"+a+"/"+b+"/"+files[n], dest_dir3+a+'/'+"validation"+'/'+b+'/'+files[n])
                        shutil.move(source_dir+"/"+a+"/"+b+"/"+files[n], dest_dir4+"validation"+'/'+a+'/'+files[n])
                    print("Total "+str(int(total_files/5))+" moved to testing folders from "+source_dir+"/"+a+"/"+b)
					
                    #move train images to targeted folder
                    for n in train_image_nums:
                        shutil.copy(source_dir+"/"+a+"/"+b+"/"+files[n], dest_dir1+a+'/'+"train"+'/'+b+'/'+files[n])
                        shutil.copy(source_dir+"/"+a+"/"+b+"/"+files[n], dest_dir2+"train"+'/'+a+'/'+files[n])
                        shutil.copy(source_dir+"/"+a+"/"+b+"/"+files[n], dest_dir3+a+'/'+"train"+'/'+b+'/'+files[n])
                        shutil.move(source_dir+"/"+a+"/"+b+"/"+files[n], dest_dir4+"train"+'/'+a+'/'+files[n])
                    print("Total "+str((total_files-int(total_files/5)))+" moved to training folders from "+source_dir+"/"+a+"/"+b)					 
    print("done!")
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source','-s', dest='source', required=True, help='REQUIRED: source type original/augmented')
    args = parser.parse_args()
    update( args.source )
