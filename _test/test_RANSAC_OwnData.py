"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#General Imports
import open3d as o3d
import copy
import numpy as np
import os
import time

BASE_DIR = os.getcwd()
# print(BASE_DIR)

from misc.dataloader import dataset_loader
from torch.utils.data import DataLoader

# h5 file_reader
from h5_files.h5_writer import append_h5, write_h5_result

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
    
def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    # pcd_down = pcd
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(templ_array, src_array, voxel_size):
    # print("\n:: Load two point clouds and disturb initial pose.")
    
    # template = r.h5reader(file_loc,"template")
    # source_ = r.h5reader(file_loc,"source")
    # gt = r.h5reader(file_loc,"transformation")
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(templ_array)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src_array)
    
    # draw_registration_result(source, target, np.identity(4),name='Original Template & Source')

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    return source, target, source_down, target_down, source_fpfh, target_fpfh

#Info on how this works: http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    Max_it = 1000000
    #Info: http://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.registration_ransac_based_on_feature_matching.html
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(Max_it, 0.999))
    return result

def test_one_epoch(test_loader, voxel_size, DIR):
    # Parameters based on "Choi_Robust_Reconstruction_of_2015_CVPR_paper"
    count = 0
    for i, data in enumerate(test_loader):
        
        template_, source_, igt_ = data
    
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
            template_[0,:,:], source_[0,:,:], voxel_size)
        
        # Plot downsampled clouds (open3D)
        # draw_registration_result(source_down, target_down, np.identity(4),
        #                          name='Original Template & Source (downsampled)')
        
        #Ransac
        start = time.time()
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        # print("===RANSAC Results===")
        registration_time = time.time() - start
        # print("RANSAC registration took %.3f sec.\n" % registration_time)
        # print(result_ransac)
        
        # Plot results (open3D)
        draw_registration_result(source, target, result_ransac.transformation, 
                                  name='Transformed Source (RANSAC) & Template')
        
        # Write result of transformation to run.log file
        append_h5(DIR,'Test',result_ransac.transformation)
        count = count + 1
    
    return registration_time/count
        
def main(h5_file_loc, object_name,voxel_size=0.01,zero_mean=True):
    
    # Create file for saving results
    DIR = write_h5_result(h5_file_loc,"RANSAC",np.zeros((0,4,4)),FolderName = "results/RANSAC/"+object_name)
    
    # Create dataset from given location with chosen parameters
    dataset = dataset_loader(h5_file_loc,zero_mean=zero_mean,normals=False)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Execute registration
    reg_time = test_one_epoch(test_loader, voxel_size, DIR)
    
    return reg_time
if __name__ == '__main__':
    main()