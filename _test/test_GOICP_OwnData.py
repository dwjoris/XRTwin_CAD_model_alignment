"""
=============================================================================
-----------------------------------CREDITS-----------------------------------
=============================================================================

GO-ICP Code by aalavandhaann
Link: https://github.com/aalavandhaann/go-icp_cython

Changes/additions by Menthy Denayer (2023)

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

# GO-ICP imports
from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE
import numpy as np
import time

import open3d as o3d #Open3D for visualisation
import os            #Os import 

# Dataloader
from misc.dataloader import dataset_loader
from torch.utils.data import DataLoader

# h5 file_reader
from h5_files.h5_writer import append_h5, write_h5_result

# Directory
BASE_DIR = os.getcwd() #Parent folder -> Thesis
# print(BASE_DIR)

"""
=============================================================================
-------------------------------------CODE------------------------------------
=============================================================================
"""

def loadPointCloud(pointcloud):
    pcloud = np.asarray(pointcloud[0]);
    plist = pcloud.tolist();
    p3dlist = [];
    for x,y,z in plist:
        pt = POINT3D(x,y,z);
        p3dlist.append(pt);
    return pcloud.shape[0], p3dlist, pcloud;

def plot_point_clouds(template_points,transf_src_points,name):
    template = o3d.geometry.PointCloud()
    transf_src = o3d.geometry.PointCloud()
    template.points = o3d.utility.Vector3dVector(template_points)
    transf_src.points = o3d.utility.Vector3dVector(transf_src_points)
    template.paint_uniform_color([0,0,1])
    transf_src.paint_uniform_color([0,1,0])
    o3d.visualization.draw_geometries([transf_src,template],window_name=name)

def test_one_epoch(testloader,goicp,rNode,tNode,DIR):
  
    # Create set of points for template/source
    a_points = [POINT3D(0.0, 0.0, 0.0), POINT3D(0.5, 1.0, 0.0), POINT3D(1.0, 0.0, 0.0)];
    b_points = [POINT3D(0.0, 0.0, 0.0), POINT3D(1.0, 0.5, 0.0), POINT3D(1.0, -0.5, 0.0)];
    
    count = 0
    
    for i, data in enumerate(testloader):
        template_, source_, igt_ = data
        
        # nmb points, go-icp format, dataset in list
        Nm, a_points, np_a_points = loadPointCloud(template_); 
        Nd, b_points, np_b_points = loadPointCloud(source_);
        
        # apply GOICP
        goicp.loadModelAndData(Nm, a_points, Nd, b_points);
        #LESS DT Size = LESS TIME CONSUMPTION = HIGHER ERROR
        goicp.setDTSizeAndFactor(300, 2.0);
        goicp.setInitNodeRot(rNode);
        goicp.setInitNodeTrans(tNode);
        
        start = time.time();
        print("\nBuilding Distance Transform...");
        goicp.BuildDT();
        print("\nREGISTERING....");
        goicp.Register();
        end = time.time();
        total_time = end-start
        print('\nTOTAL TIME : ', total_time);
        optR = np.array(goicp.optimalRotation());
        optT = goicp.optimalTranslation();
        optT.append(1.0);
        optT = np.array(optT);
        
        transform = np.eye((4));
        transform[:3, :3] = optR;
        transform[:,3] = optT;
        
        # print(np_b_points.shape, np.ones((Nd, 1)).shape);
        
        #Now transform the data mesh to fit the model mesh
        transform_model_points = (transform.dot(np.hstack((np_b_points, np.ones((Nd, 1)))).T)).T;
        transform_model_points = transform_model_points[:,:3];
        
        # Save transformed points into .ply file
        # PLY_FILE_HEADER = "ply\nformat ascii 1.0\ncomment PYTHON generated\nelement vertex %s\nproperty float x\nproperty float y\nproperty float z\nend_header"%(Nd);
        # np.savetxt(BASE_DIR+'/_test/GO-ICP/Base-Top_Plate2.ply', transform_model_points, header = PLY_FILE_HEADER, comments='');
        
        # Plot results (open3D)
        # plot_point_clouds(np_a_points,transform_model_points,
        #                   name='Transformed Source (GO-ICP) & Template')
        
        # Write result of transformation to h5 file
        # print(transform)
        append_h5(DIR,'Test',transform)
        count = count + 1
    
    return total_time/count

def main(h5_file_loc,object_name,MSEThresh=0.00001,trimFraction=0.0001,zero_mean=False, voxel_size=0):
    
    # Create file for saving results
    DIR = write_h5_result(h5_file_loc,"GO-ICP", voxel_size, 
                          np.zeros((0,4,4)),FolderName = "results/GO-ICP/"+object_name)
    
    goicp = GoICP();     # Create GoICP class
    rNode = ROTNODE();   # Create translation nodes
    tNode = TRANSNODE(); # Create rotation nodes
    
    # Node limits
    rNode.a = -3.1416;
    rNode.b = -3.1416;
    rNode.c = -3.1416;
    rNode.w = 6.2832;
     
    tNode.x = -0.5;
    tNode.y = -0.5;
    tNode.z = -0.5;
    tNode.w = 1.0;
    
    goicp.MSEThresh = MSEThresh;
    goicp.trimFraction = trimFraction;
      
    if(goicp.trimFraction < 0.001):
        goicp.doTrim = False;
    
    # Create dataset from given location with chosen parameters
    dataset = dataset_loader(h5_file_loc,zero_mean=zero_mean,normals=False, voxel_size = voxel_size)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Execute registration
    reg_time = test_one_epoch(test_loader,goicp,rNode,tNode,DIR)
    
    return reg_time

if __name__ == '__main__':
    main()

