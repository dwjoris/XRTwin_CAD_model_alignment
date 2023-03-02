import open3d as o3d

#Link: https://stackoverflow.com/questions/28170623/how-to-read-hdf5-files-in-python

import h5py
# filename = "C:/Users/menth/Downloads/test_error6.hdf5"
# filename = "C:/Users/menth/Documents/Python Scripts/thesis/h5_files/output/test_STL/test_STL.hdf5"
filename = "C:/Users/menth/Documents/Python Scripts/thesis/h5_files/output/experiments/Shaft_New_1_BB_1.hdf5"
# filename = "C:/Users/menth/Documents/Python Scripts/Thesis/toolboxes/learning3d/data/modelnet40_ply_hdf5_2048/ply_data_train1.h5"
# filename = "C:/Users/menth/Documents/Python Scripts/Thesis/toolboxes/learning3d/data/modelnet40_ply_hdf5_2048"
#filename = "C:/Users/menth/Documents/Python Scripts/Thesis/toolboxes/learning3d/checkpoints/exp_pnlk/errors/EP200/train_error1.hdf5"

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[3]
    
    print("Current Key: %s" % a_group_key)
    print("File content for selected key: %s" % f[a_group_key])

    # get the object type for a_group_key: usually group or dataset
    print("Object type for key: %s" % type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    #data = list(f[a_group_key])

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    #data = list(f[a_group_key])
    # preferred methods to get dataset values:
    ds_obj = f[a_group_key]      # returns as a h5py dataset object
    ds_arr = f[a_group_key][()]  # returns as a numpy array
    # ds_arr2 = f['labels'][()]  # returns as a numpy array
    ds_arr2 = f['source'][()]  # returns as a numpy array
    #gt = f['gt'][()]
    #T = f['est_T'][()]
    
    #print(ds_arr[:][:][:])
    
    if(a_group_key == "template"):
        for i in range(100):
        #added -> visualize
            source_ = o3d.geometry.PointCloud()
            source_.points = o3d.utility.Vector3dVector(ds_arr2[i][:][:])
            source_.paint_uniform_color([1,0,0])
            # template_ = o3d.geometry.PointCloud()
            # template_.points = o3d.utility.Vector3dVector(ds_arr[i][:][:])
            # template_.paint_uniform_color([0,1,0])
            # template_2 = o3d.geometry.PointCloud()
            # template_2.points = o3d.utility.Vector3dVector(ds_arr[i+1][:][:])
            # template_2.paint_uniform_color([1,0,0])
            o3d.visualization.draw_geometries([source_])
            # print(ds_arr2[i])