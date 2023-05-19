"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



test ICP OwnData

Run ICP, with given parameters, for given .h5 file

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth

Output:
    - .hdf5 file with estimated transformation

Credits: 
    ICP as part of the Open3D library

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#General Imports
import open3d as o3d
import copy
import numpy as np
import time
import os

BASE_DIR = os.getcwd() #Parent folder -> Thesis
# print(BASE_DIR)

from misc.dataloader import dataset_loader
from torch.utils.data import DataLoader

# h5 writer
from h5_files.h5_writer import write_h5_result, append_h5

"""
=============================================================================
-------------------ICP CLASS (for changing basic parameters)-----------------
=============================================================================
"""

class ICP():
    def __init__(self, voxel_size = 0.01, refine = True):
        self.feature_radius_factor = 5
        self.normal_radius_factor = 2
        self.normal_nn = 30
        self.feature_nn = 100
        self.distance_threshold_factor = 0.4
        self.nmb_it = 100000
        self.voxel_size = voxel_size
        self.refine = refine
        

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

def draw_registration_result(source, target, transformation,name):
    transf_source_temp = copy.deepcopy(source)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1,0,0])
    target_temp.paint_uniform_color([0,0,1])
    transf_source_temp.paint_uniform_color([0,1,0])
    transf_source_temp.transform(transformation)
    
    o3d.visualization.draw_geometries([transf_source_temp,target_temp],
                                      window_name=name)
    
def preprocess_point_cloud(pcd, icp):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(icp.voxel_size)

    radius_normal = icp.voxel_size * icp.normal_radius_factor
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=icp.normal_nn))

    radius_feature = icp.voxel_size * icp.feature_radius_factor
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=icp.feature_nn))
    return pcd_down, pcd_fpfh

def prepare_dataset(templ_array, src_array, transfo_array, icp):
    # print(":: Load point clouds and initial transformation.")
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(templ_array)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src_array)
    
    # draw_registration_result(source, target, np.identity(4),name='Original Template & Source')

    source_down, source_fpfh = preprocess_point_cloud(source, icp)
    target_down, target_fpfh = preprocess_point_cloud(target, icp)
    
    return source, target, transfo_array, source_down, target_down, source_fpfh, target_fpfh

def refine_registration(source, target, T_est, source_down, target_down, source_fpfh, target_fpfh, icp):
    distance_threshold = icp.voxel_size * icp.distance_threshold_factor
    Max_it = icp.nmb_it
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    # print(source)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, T_est,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=Max_it)) #o3d.pipelines.registration.TransformationEstimationPointToPlane()
    
    results_PlanetoPlane = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, T_est,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=Max_it))
    return result, results_PlanetoPlane

def test_one_epoch(test_loader, icp, DIR):
    count = 0
    tot_reg_time = 0
    for i, data in enumerate(test_loader):
        if(icp.refine):
            template_, source_, T_est = data
            source, target, T_est, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
                template_[0,:,:], source_[0,:,:], T_est[0,:,:], icp)
        else:
            template_, source_, _ = data
            T_est = np.identity(4)
            source, target, T_est, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
                template_[0,:,:], source_[0,:,:], T_est, icp)
        
        # Plot downsampled PCRs (open3D)
        # draw_registration_result(source_down, target_down, T_est,
        #                           name='Original Template & Source (downsampled)')
        
        #ICP
        start = time.time()
        result_icp, result_PtP = refine_registration(source, target, T_est, source_down, target_down, 
                                                     source_fpfh, target_fpfh, icp)
        
        # print("===ICP-Ref. Results===")
        registration_time = (time.time() - start)
        # print("ICP registration took %.3f sec.\n" % registration_time)
        # print(result_icp)
        # print(result_PtP)
        
        # Visualise result
        # draw_registration_result(source, target, result_icp.transformation,
        #                           name='Transformed Source (ICP) & Template')
        # draw_registration_result(source, target, result_PtP.transformation)
        
        # Write result of transformation to h5 file
        count = count + 1
        tot_reg_time = tot_reg_time + registration_time
        append_h5(DIR,'Test',result_icp.transformation)
        
    return tot_reg_time/count

def main(h5_file_loc, object_name, refine = True, voxel_size=0.01,zero_mean=True):
    
    # Create file for saving results
    DIR = write_h5_result(h5_file_loc,"ICP", voxel_size, np.zeros((0,4,4)),FolderName = "results/ICP/"+object_name)
    
    # Create dataset from given location with chosen parameters
    dataset = dataset_loader(h5_file_loc,zero_mean=zero_mean, normals=False,Test=refine)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialize ICP variables
    icp = ICP(voxel_size,refine)
    
    # Execute registration
    reg_time = test_one_epoch(test_loader, icp, DIR)

    return reg_time

if __name__ == '__main__':
    main()