"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



dataloader

Loads in the .hdf5 file and returns a dataset loader object with the desired
parameters to loop over all objects inside.

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth
        o Estimated Transformation

Output:
    - dataloader object
"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""
import torch
import numpy as np
import open3d as o3d
# from functions.file_reader import h5reader
# from torch.utils.data import DataLoader
import h5py

"""
=============================================================================
-----------------------------------CLASS-------------------------------------
=============================================================================
"""

class dataset_loader():
    """
    Create class to load .hdf5 files with desired parameters
    
    Parameters
    ----------
    file_loc                : String            // Location of .hdf5 file
    zero_mean               : Boolean           // Center point clouds on origin and adapt ground truth accordingly
    normals                 : Boolean           // Normals included in point cloud
    voxel_size              : float             // voxel size for downsampling 
    Test                    : Boolean           // Whehter .hdf5 file already includes an estimated transformation
    
    Returns
    ----------
    dataset_loader object   : data loader object      
    """

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
        """
        Create class to load .hdf5 files with desired parameters
        
        Parameters
        ----------
        pcd_array                : 1xNx6 numpy array        // list of point cloud points
        
        Returns
        ----------
        pcd_down_array           : 1xNx6 numpy array        // list of point cloud points (downsampled)     
        """
        
        """
        :: WARNING :: 
        :: Downsampling point cloud only implemented when 1 object in .hdf5 file, not for training sets
        """
        
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


"""
=============================================================================
---------------------------------FUNCTIONS-----------------------------------
=============================================================================
"""

def remove_mean(pcd_array):
    """
    Remove mean from point cloud
    
    Parameters
    ----------
    pcd_array                : 1xNx6 numpy array        // list of point cloud points
    
    Returns
    ----------
    pcd_array_no_mean       : 1xNx6 numpy array         // list of point cloud points (centered)  
    pcd_mean                : 1x3 numpy array           // computed mean of array
    """
    
    pcd_mean = np.mean(pcd_array,1)[:,0:3]
    # print(pcd_mean)
    pcd_array_no_mean = pcd_array
    pcd_array_no_mean[:,:,0:3] = pcd_array_no_mean[:,:,0:3] - pcd_mean
    return pcd_array_no_mean, pcd_mean

def remove_mean_transformation(transformation_array,mean):
    """
    Remove mean from translation vector in ground truth
    
    Parameters
    ----------
    transformation_array            : Nx4x4 numpy array     // list of ground truth solutions
    mean                            : 1x3 numpy array       // computed mean of point cloud
    
    Returns
    ----------
    transformation_array_no_mean    : Nx4x4 numpy array     // list of updated ground truth solutions
    """
        
    transformation_array_no_mean = transformation_array
    nmb_transf = transformation_array.shape[0]
    
    for i in range(nmb_transf):
        transformation_array_no_mean[i,0:3,3] = transformation_array_no_mean[i,0:3,3] - mean
    
    return transformation_array_no_mean

def h5reader(file_dict,key):
    """
    Read h5 file from directory with given key
    
    Parameters
    ----------
    file_dict               : String                // Location of .hdf5 file
    key                     : String                // Key to read from .hdf5 file
    
    Returns
    ----------
    dataset                 : NxMxP numpy array     // data read from .hdf5 file
    """
    
    """
    Source: https://stackoverflow.com/questions/28170623/how-to-read-hdf5-files-in-python  
    """
    
    with h5py.File(file_dict,"r") as f:
        if(key in f.keys()):
            dataset = f[key][()]
        else:
            raise Exception("Key not included in %s" % f.keys())
    return dataset


def h5file_to_torch(file_loc, zero_mean, T_est = False):
    """
    Turn basic .hdf5 file, containing template, source, ground truth and estimated transformation into tensor
    
    Parameters
    ----------
    file_loc                    : String                // Location of .hdf5 file
    zero_mean                   : Boolean               // Center point clouds
    T_est                       : Boolean               // Whehter .hdf5 file contains registration result
    
    Returns
    ----------
    templ_tensor                : 1xNx6 torch tensor    // list of template points
    src_tensor                  : 1xMx6 torch tensor    // list of source points
    gt_tensor                   : 1x4x4 torch tensor    // first ground truth solution
    gt_symm_tensor              : Px4x4 torch tensor    // list of all ground truth solutions
    """
    
    dataset = dataset_loader(file_loc, zero_mean, Test = T_est)
    templ_array = dataset.template
    src_array = dataset.source
    gt_array = dataset.transformation
    gt_symm_array = dataset.transformation_all
    
    templ_tensor = torch.tensor(templ_array)
    src_tensor = torch.tensor(src_array)
    gt_tensor = torch.tensor(gt_array)
    gt_symm_tensor = torch.tensor(gt_symm_array)
    
    if(T_est):
        transfo_array = dataset.estimated_transform
        transfo_tensor = torch.tensor(transfo_array,dtype=torch.float64)
        return templ_tensor, src_tensor, gt_tensor, gt_symm_tensor, transfo_tensor
    
    return templ_tensor, src_tensor, gt_tensor, gt_symm_tensor