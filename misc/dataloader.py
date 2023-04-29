
import torch
import numpy as np
import open3d as o3d
from h5_files.file_reader import h5reader
from torch.utils.data import DataLoader

class dataset_loader():
    def __init__(self,file_loc,zero_mean = False, normals = False, voxel_size = 0, Test = False):
        self.file_loc = file_loc
        self.template = h5reader(file_loc,'template')
        self.source = h5reader(file_loc,'source')
        self.transformation = h5reader(file_loc,'transformation')
        self.transformation_all = h5reader(file_loc,'transformation_all')
        self.normals = normals
        self.voxel_size = voxel_size
        self.Test = Test
        
        if(Test):
            self.estimated_transform = h5reader(file_loc,'Test')
        
        if(zero_mean):
            self.template,self.template_mean = remove_mean(self.template)
            self.source,self.source_mean = remove_mean(self.source)
            self.transformation = remove_mean_transformation(self.transformation, self.source_mean)
            self.transformation_all = remove_mean_transformation(self.transformation_all, self.source_mean)

        if(voxel_size != 0):
            self.template = self.downsample_pcd(self.template)
            self.source = self.downsample_pcd(self.source)
        
        if(normals):
            self.DIM = 6
        else:
            self.DIM = 3
        
    def __len__(self):
        return self.template.shape[0] 
    
    def __getitem__(self,index):
        if(self.Test):
            return torch.tensor(self.template[index,:,:self.DIM]).float(), \
                torch.tensor(self.source[index,:,:self.DIM]).float(), \
                torch.tensor(self.estimated_transform[index]).float() 
        else:
            return torch.tensor(self.template[index,:,:self.DIM]).float(),\
                torch.tensor(self.source[index,:,:self.DIM]).float(), \
                torch.tensor(self.transformation[index]).float() 
    
    def downsample_pcd(self, pcd_array):
        # Turn point cloud array into point cloud
        r = 2 * self.voxel_size
        neigh_max = 30
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_array[0,:,0:3])
        
        # Downsample with given voxelsize
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        pcd_down_points = np.asarray(pcd_down.points)
        
        if(self.normals):
            pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=r, max_nn=neigh_max))
            
            pcd_normals = np.asarray(pcd_down.normals)
            pcd_down_points = np.concatenate((pcd_down_points,pcd_normals),1)
        
        pcd_down_array =  np.expand_dims(pcd_down_points, 0)
        return pcd_down_array

def remove_mean(pcd_array):
    pcd_mean = np.mean(pcd_array,1)[:,0:3]
    # print(pcd_mean)
    pcd_array_no_mean = pcd_array
    pcd_array_no_mean[:,:,0:3] = pcd_array_no_mean[:,:,0:3] - pcd_mean
    return pcd_array_no_mean, pcd_mean

def remove_mean_transformation(transformation_array,mean):
    transformation_array_no_mean = transformation_array
    nmb_transf = transformation_array.shape[0]
    
    for i in range(nmb_transf):
        transformation_array_no_mean[i,0:3,3] = transformation_array_no_mean[i,0:3,3] - mean
    
    return transformation_array_no_mean

# file_loc1 = "C:/Users/menth/Documents/Python Scripts/Thesis/h5_files/output/experiments/BB_Effect/BB_1.0/Base-Top_Plate/Base-Top_Plate_2_BB_1.0_Normals.hdf5"
# file_loc2 = "C:/Users/menth/Documents/Python Scripts/Thesis/h5_files/output/experiments/BB_Effect/BB_1.2/Base-Top_Plate/Base-Top_Plate_2_BB_1.2_Normals.hdf5"
# file_loc3 = "C:/Users/menth/Documents/Python Scripts/Thesis/h5_files/output/experiments/BB_Effect/BB_1.4/Base-Top_Plate/Base-Top_Plate_2_BB_1.4_Normals.hdf5"
# file_loc4 = "C:/Users/menth/Documents/Python Scripts/Thesis/h5_files/output/experiments/BB_Effect/BB_1.6/Base-Top_Plate/Base-Top_Plate_2_BB_1.6_Normals.hdf5"
# file_loc5 = "C:/Users/menth/Documents/Python Scripts/Thesis/h5_files/output/experiments/BB_Effect/BB_1.8/Base-Top_Plate/Base-Top_Plate_2_BB_1.8_Normals.hdf5"

# file_locs = [file_loc1,file_loc2,file_loc3,file_loc4,file_loc5];

# for file in file_locs:

#     dataset = dataset_loader(file,zero_mean=False,normals=True, Test=False,voxel_size=0.002)
#     test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
#     for i,data in enumerate(test_loader):
#         templ, source, gt = data
    
        
#     templ_pc = o3d.geometry.PointCloud()
#     templ_pc.points = o3d.utility.Vector3dVector(source[0][:,0:3])
#     templ_pc.normals = o3d.utility.Vector3dVector(source[0][:,3:6])
#     templ_pc.paint_uniform_color([0,0,1])
#     o3d.visualization.draw_geometries([templ_pc])