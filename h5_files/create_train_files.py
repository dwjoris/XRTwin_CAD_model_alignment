#PointNetLK: training around random axis over 0->45° and translation between [0;0.8]
# + PC template normalized between [0,1] + 400 Epochs for good results
# + training on partial data by also using estimate of camera position 
#(removing points from template too)

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#General imports
import os
import open3d as o3d
import torch
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle

os.chdir("C:/Users/menth/Documents/Python Scripts/Thesis")

#h5_files imports
import h5_files.h5_writer as w
import h5_files.file_reader as r

"""
=============================================================================
--------------------------------CAD FILES DIR--------------------------------
=============================================================================
"""

#CAD Files directory
DIR = "C:/Users/menth/Documents/Python Scripts/Thesis/datasets/CAD/Limited"

"""
=============================================================================
-------------------------------FILE PARAMETERS-------------------------------
=============================================================================
"""

#Parameters
# 9843 total train files in ModelNet40 
#/2 (half the classes usually used for training)
#/6 objects in our set
# ~= 820 it/object training
# ~= 1/4 testing
# ~= 205 it/object testing
Number_Iter_train = 820                                           #Nmb of different itr of same object
Number_Iter_test = 5                                            #Nmb of different itr of same object
mu_noise = 0                                                      #Mean value for noise
sigma_noise_max = 0.01                                            #Std deviation value for noise
sigma_noise_bound = 0.05                                          #Values outside [-0.05,0.05] clipped

Nmb_points = 2048                                                 #Nmb of points in template PC
Keep = 0.7                                                        #% of points to keep after partial

Nmb_points_partial = int(Keep*Nmb_points)                         #Nmb of points to keep after partial

"""
=============================================================================
----------------------------------FUNCTIONS----------------------------------
=============================================================================
"""
def get_transformations(igt):
    R_ba = igt[0, 0:3, 0:3]                                # Ps = R_ba * Pt
    translation_ba = igt[0, 0:3, 3]                        # Ps = Pt + t_ba
    R_ab = np.transpose(R_ba)                              # Pt = R_ab * Ps
    translation_ab = -np.matmul(R_ab, translation_ba)      # Pt = Ps + t_ab
    translation_ab = np.expand_dims(translation_ab,-1)
    return np.expand_dims(R_ab,0), np.expand_dims(translation_ab,0)

def generate_ground_truth():
    #Random translation between [-0.5,0.5]
    t_array = np.random.rand(3)-0.5                              #Translation vector, Ps = Pt + t
    R_array = w.random_rotation()                                #Rotation matrix,    Ps = R*Pt

    gt_array = w.homogenous_transfo(R_array,t_array)             #Transform R,t to T matrix
    gt_array = np.expand_dims(gt_array,0)                        #Correct dimension: 1x4x4    
    return gt_array,R_array, t_array

def turn_to_partial(src_array, Nmb_points_partial):
    src_array, mask = w.farthest_subsample_points(src_array[0][:][:], Nmb_points_partial) 
    return src_array

def create_one_set(template_array, normals, noise, partial):
    #Generate ground truth and apply transformation
    gt_array,R_array, t_array = generate_ground_truth()
    src_array = w.apply_transfo(template_array, R_array, t_array, normals)

    #Add Gaussian noise 
    if(noise):
        src_array = w.add_Gaussian_Noise(mu_noise, sigma_noise_max, 
                                      src_array,sigma_noise_bound)
    
    #Turn to partial PC
    if(partial):
        src_array = turn_to_partial(src_array, Nmb_points_partial)
        src_array = np.expand_dims(src_array,0)
    
    return src_array, gt_array

def create_template(path, normals = False):
    # Create template file
    file_path = os.path.join(DIR, path)
    template_array = r.read_stl(file_path,Nmb_points, normals)
    template_array = np.expand_dims(template_array,0)
    return template_array

def create_set(DIR, Number_Iter, partial = False, noise = False, normals = False, sep = False):
    #Initialize lists to store values
    total_template_list = []
    total_source_list = []
    total_Rotation_list = []
    total_translation_list = []
    total_gt_list = []
    
    for i,path in enumerate(tqdm(os.listdir(DIR))):
        
        # check if current path is a file
        if os.path.isfile(os.path.join(DIR, path)):
            
            #Create template
            template_array = create_template(path, normals)
            
            #Loop over nmb of iterations
            for it in range(Number_Iter):
                
                #Create source and ground truth
                src_array, gt_array = create_one_set(template_array, normals, noise, partial)
                
                #Turn to list for speed
                tmplt_list = template_array[0].tolist()
                src_list = src_array[0].tolist()
                gt_list = gt_array[0].tolist()
                
                #Save data into lists
                total_template_list.append(tmplt_list)
                total_source_list.append(src_list)

                if(sep):
                    Rab, tab = get_transformations(gt_array)
                    total_Rotation_list.append(Rab[0].tolist())
                    total_translation_list.append(tab[0].tolist())
                else:
                    total_gt_list.append(gt_list)
    
    #Shuffle data
    if(sep):
        total_template_list, total_source_list, total_Rotation_list, total_translation_list = shuffle(
            total_template_list, total_source_list, total_Rotation_list, total_translation_list ,random_state=0)
        
        return np.array(total_template_list), np.array(total_source_list), np.array(total_Rotation_list), np.array(total_translation_list)
    
    else:
        total_template_list, total_source_list, total_gt_list = shuffle(
            total_template_list, total_source_list, total_gt_list, random_state=0)
        
    return np.array(total_template_list), np.array(total_source_list), np.array(total_gt_list)

def create_set_labeled(DIR,Number_Iter):
    #Initialize lists to store data
    total_template_list = []
    labels = []
    for i,path in enumerate(tqdm(os.listdir(DIR))):
        
        # check if current path is a file
        if os.path.isfile(os.path.join(DIR, path)):
            
            #Create template
            template_array = create_template(path)
            
            for it in range(Number_Iter):
                labels.append([i])
                #Random translation between [-0.5,0.5]
                template_array, gt_array = create_one_set(template_array, False, False, False)
                
                #Turn to list for speed
                tmplt_list = template_array[0].tolist()
                
                #Save data into lists
                total_template_list.append(tmplt_list)
    
    template_list, labels = shuffle(total_template_list, labels,random_state=0)
    
    return template_list, labels

def name(Noise, Partial, Labeled, Normals):
    name_train = "ply_data_train_" + str(Number_Iter_train) + "IT"
    name_test = "ply_data_test_" + str(Number_Iter_test) + "IT"
    add = ""
    
    if(Noise == False and Partial == False):
        add = add + "_normal" 
    if(Noise == True):
        add = add + "_noisy_" + str(sigma_noise_max)
    if(Partial == True):
        add = add + "_partial_" + str(Keep)
    if(Labeled == True):
        add = add + "_labeled"
    if(Normals == True):
        add = add + "_normalinfo"
    
    name_train = name_train + add
    name_test = name_test + add
    
    return name_train, name_test

"""
=============================================================================
----------------------------------EXECUTION----------------------------------
=============================================================================
"""

def main():
    Noise = False
    Partial = True
    Normals = False
    
    # Unlabeled data (PointNetLK, RPMNet, ROPNet,...)
    name_train, name_test = name(Noise, Partial, False, Normals) # correct naming convention
    
    # train dataset
    # template_list_train, source_list_train, gt_list_train = create_set(DIR, 
                                                            # Number_Iter_train, Partial, Noise, Normals)
    # w.write_h5(name_train, template_list_train, source_list_train, gt_list_train, "files_train/")
    
    # test dataset
    template_list_test, source_list_test, gt_list_test = create_set(DIR, 
                                                            Number_Iter_test, Partial, Noise, Normals)
      
    w.write_h5(name_test, template_list_test, source_list_test, gt_list_test, "files_test/")
    
    # Visualize train/test data
    # for i in range(0,Number_Iter_test):
    #     # r.show_open3d(template_list_train[:,:,0:3], source_list_train[:,:,0:3], index = i)
    #     r.show_open3d(template_list_test[:,:,0:3], source_list_test[:,:,0:3], index = i)
    #     r.show_open3d(template_list_train[:,:,0:3], source_list_train[:,:,0:3], index = i)
    
    # # labeled data (PointNetLK, RPMNet, ROPNet,...)
    # name_train_labeled, name_test_labeled = name(Noise, Partial, True, Normals)

    # data_train, labels_train = create_set_labeled(DIR, Number_Iter_train)
    # w.write_h5_labeled(name_train_labeled, data_train, labels_train, "files_train/")
    
    # data_test, labels_test = create_set_labeled(DIR, Number_Iter_test)
    # # print(np.array(data_test).shape)
    # w.write_h5_labeled(name_test_labeled, data_test, labels_test, "files_test/")

    # R/t split data
    # template_list_train, source_list_train, R_list_train, t_list_train = create_set(DIR, 
    #                                                         Number_Iter_train, Partial, Noise, True)
    
    # w.write_h5_sep(name_train, template_list_train, source_list_train, R_list_train, t_list_train
    #             , "files_train/")
    
    # template_list_test, source_list_test, R_list_test, t_list_test = create_set(DIR, 
    #                                                         Number_Iter_test, Partial, Noise, True)
    
    # w.write_h5_sep(name_test, template_list_test, source_list_test, R_list_test, t_list_test
    #                 , "files_test/")
    
if __name__ == '__main__':
    main()