"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



PCL Post Processing

Post process raw point cloud data into txt file for registration. Post processing
includes: 
    - removing floor (RANSAC plane fitting)
    - segmenting point cloud (DBSCAN)
    - extract relevant segmented part
    - denoising point cloud (remove statistical outliers)
    - using bounding box to remove point cloud 
    - potentially add normals information

Inputs:
    - .ply file         (Source, RealSense)

Output:
    - .txt file containing:
        o Source (w/ normals)

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
import math

from h5_files.h5_writer import uniquify
"""
=============================================================================
-------------------------------ENTER FILE NAME-------------------------------
=============================================================================
"""

# PC_file = "Base-Top_Plate_5.ply"
# PC_file = "Round-Peg_4.ply"

global color_red
global color_blue
global color_yellow
global color_gray

color_red = [0.87843137, 0.16078431, 0.11372549]    # E0291D
color_blue = [0.11372549, 0.14509804, 0.87843137]   # 1D25E0
color_yellow = [1., 0.90196078, 0.]                 # FFE600
color_gray = [0.6, 0.6, 0.6]

"""
=============================================================================
----------------------------------FUNCTIONS----------------------------------
=============================================================================
"""

def draw_template(pcd, src_mean, normal_vector, Y_angle, CAD_file):
    # Read Mesh file
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    
    # Angles
    Plane_Normal_Norm = np.linalg.norm(np.array(normal_vector))

    X_angle = math.acos(normal_vector[1]/Plane_Normal_Norm)*180/math.pi
    
    # frame of source
    mesh_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=src_mean)
    R = mesh_frame2.get_rotation_matrix_from_xyz((X_angle*math.pi/180,Y_angle,0))
    mesh_frame2.rotate(R, center=src_mean)
    
    template_filename = os.path.join(os.getcwd() + "/datasets/CAD/Original/" + CAD_file + ".stl")
    template_mesh = o3d.io.read_triangle_mesh(template_filename)  
    
    # Sample Mesh uniformly
    templ_nmb_points = 1024*5
    templ_pcd = template_mesh.sample_points_uniformly(number_of_points=templ_nmb_points) 
    templ_array = np.asarray(templ_pcd.points)/100
    templ_mean = np.mean(templ_array,0)
    
    templ_array = np.add(templ_array,-templ_mean)
    
    templ_pcd.points = o3d.utility.Vector3dVector(templ_array)
    
    o3d.visualization.draw_geometries([mesh_frame, mesh_frame2, pcd, templ_pcd])
    # Transfrorm template PC
    templ_pcd.rotate(R,center=[0,0,0])
    templ_pcd.translate(src_mean+np.asarray([0,0.005,-0.03]))
    
    o3d.visualization.draw_geometries([mesh_frame, mesh_frame2, pcd, templ_pcd])
    # o3d.visualization.draw_geometries([mesh_frame, mesh_frame2])

def estimate_normals(cloud, r = 0.1, neigh_max = 30):
    print(":: Estimating normals...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=r, max_nn=neigh_max))
    
    pcd.paint_uniform_color(color_red)
    o3d.visualization.draw_geometries([pcd])
    
    return pcd

# Guided filter created by aipiano, LINK: https://github.com/aipiano/guided-filter-point-cloud-denoise

def guided_filter(pcd, radius, epsilon):
    print(":: Denoising pointcloud...")
    new_pcd = o3d.geometry.PointCloud()
    
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points_copy = np.array(pcd.points)
    points = np.asarray(pcd.points)
    num_points = len(pcd.points)

    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if k < 3:
            continue

        neighbors = points[idx, :]
        mean = np.mean(neighbors, 0)
        cov = np.cov(neighbors.T)
        e = np.linalg.inv(cov + epsilon * np.eye(3))

        A = cov @ e
        b = mean - A @ mean

        points_copy[i] = A @ points[i] + b

    new_pcd.points = o3d.utility.Vector3dVector(points_copy)
    new_pcd.paint_uniform_color(color_yellow)
    pcd.paint_uniform_color(color_blue)
    o3d.visualization.draw_geometries([pcd,new_pcd])
    return new_pcd
    
def save(pcd,name,Normals=False):
    print(":: Saving file...")
    
    save = np.asarray(pcd.points)
    
    if(Normals):
        normals = np.asarray(pcd.normals)
        save = np.concatenate((save,normals),1)
        name = name + "_Normals"

    DIR = os.path.join(os.getcwd() + "/h5_files/output/processing_results/" + name + ".txt")
    DIR = uniquify(DIR)
    np.savetxt(DIR, save)

    print(":: Saving finished, saved as: ", DIR)
    return

def m_to_px(pcd, width_x, width_y):
    #Find max/min x/y values for finding conversion px <-> meters
    y_min = np.min(pcd.points[:][1])
    y_max = np.max(pcd.points[:][1])

    x_min = np.min(pcd.points[:][0])
    x_max = np.max(pcd.points[:][0])

    print(":: Min y-coordinate: ", y_min)
    print(":: Max y-coordinate: ", y_max)
    print(":: Min x-coordinate: ", x_min)
    print(":: Max x-coordinate: ", x_max)

    coef_y = (y_max-y_min)/width_x
    coef_x = (x_max-x_min)/width_y

    return coef_x, coef_y

def inside_rectangle(pcd, corners):
    print(":: Extracting points inside given rectangle...")
    
    dir1 = corners[1]-corners[0] # AB
    dir2 = corners[2]-corners[0] # AC
    dir3 = corners[3]-corners[0] # AD
    points = np.array(pcd.points)
    
    points_list = []
    for point in points:
        dirM = point-corners[0] # for point M, AM
        if all(point_inside(dir1,dir2,dir3, dirM)):
            points_list.append(point.tolist())
    
    inside_rect = o3d.geometry.PointCloud()
    # print(np.array(points_list).shape)
    inside_rect.points = o3d.utility.Vector3dVector(np.array(points_list))
    inside_rect.paint_uniform_color(color_red)
    o3d.visualization.draw_geometries([inside_rect])
    
    return inside_rect

def get_bounding_box(pcd):
    print(":: Extracting bounding boxes...")
    obb = pcd.get_oriented_bounding_box()
    obb.color = tuple(color_blue)
    
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = tuple(color_yellow)
    
    return aabb, obb

def remove_outliers(pcd, nb_neigh, std):
    print(":: Removing outliers...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neigh,std_ratio=std)
    cl.paint_uniform_color(color_blue)
    pcd.paint_uniform_color(color_yellow)
    o3d.visualization.draw_geometries([cl,pcd])
    return cl

def extract_object(pcd, labels, label_nmb):
    print(":: Extracting object with label: ",label_nmb)
    obj_points = np.array(pcd.points)[labels==label_nmb]
    obj = o3d.geometry.PointCloud()
    obj.points = o3d.utility.Vector3dVector(obj_points)
    o3d.visualization.draw_geometries([obj])
    obj.paint_uniform_color(color_red)
    return obj

def segmentation(pcd, eps_value, min_nmb_points):
    print(":: Segmenting point cloud...")
    #DBSCAN -> clustering method on outlier point cloud
    labels = np.array(pcd.cluster_dbscan(eps=eps_value, min_points=min_nmb_points))
    max_label = labels.max()

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])
    print(":: Computed " + str(max_label) + " labels")
    return pcd, labels

def plane_fitting(pcd, dist_thresh, ransac_nmb, it):
    print(":: Performing RANSAC plane fitting...")
    #RANSAC plane fitting
    #Link: https://towardsdatascience.com/how-to-automate-3d-point-cloud-segmentation-and-clustering-with-python-343c9039e4f5
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_thresh, 
                                             ransac_n=ransac_nmb, num_iterations=it)

    [a,b,c,d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    print(f":: Plane normal vector: [ {a:.4f}, {b:.4f}, {c:.4f}]")
    normal_vector = np.asarray([a,b,c])
    
    inlier = pcd.select_by_index(inliers)
    outlier = pcd.select_by_index(inliers, invert=True)

    inlier.paint_uniform_color(color_red)
    outlier.paint_uniform_color(color_gray)
    
    o3d.visualization.draw_geometries([inlier, outlier])
    
    return inlier, outlier, normal_vector

def point_inside(dir1,dir2,dir3, dirM):
    yield 0 < dirM @ dir1 < dir1 @ dir1 # @ symbol for dot product, 0 < AM * AB < AB*AB
    yield 0 < dirM @ dir2 < dir2 @ dir2 # @ symbol for dot product, 0 < AM * AC < AC*AC
    yield 0 < dirM @ dir3 < dir3 @ dir3 # @ symbol for dot product, 0 < AM * AD < AD*AD

def remove_after_z(point_cloud,d):
    print(":: Removing points with depth > ", d)
    points = np.asarray(point_cloud.points)
    new_cloud = np.array([[0,0,0]])
    num_points = points.shape[0]
    for i in range(num_points):
        #print(int(i/num_points*100)*"-")
        if(abs(points[i][2]) < d):
            #rint(points[i])
            new_points = np.expand_dims(points[i],0)
            #print(new_points)
            new_cloud = np.append(new_cloud,new_points,0)
    #print(new_cloud)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(new_cloud[1:][:])
    return cloud

def keep_rectangle(point_cloud,start_point,end_point, coef_x, coef_y, width_x, width_y):
    print(":: Removing points outside given rectangle...")
    #print("test")
    points = np.asarray(point_cloud.points)
    new_cloud = np.array([[0,0,0]])
    num_points = points.shape[0]
    
    x_max = coef_x*(end_point[0]-width_x/2)
    x_min = coef_x*(start_point[0]-width_x/2)
    y_max = coef_y*(end_point[1]-width_y/2)
    y_min = coef_y*(start_point[1]-width_y/2)
    
    for i in range(num_points):
        #print(int(i/num_points*100)*"-")
        if(points[i][0] < x_max and points[i][0] > x_min and points[i][1] < y_max and points[i][1] > y_min):
            new_points = np.expand_dims(points[i],0)
            new_cloud = np.append(new_cloud,new_points,0)
    #print(new_cloud)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(new_cloud[1:][:])  
    return cloud

"""
=============================================================================
-------------------------------POST PROCESSING-------------------------------
=============================================================================
"""

def main(PC_file, BB_factor):
    # Find file DIR
    fileDIR = os.path.join(os.getcwd() + "/h5_files/realsense/" + PC_file + ".ply")
    
    #Read Point Cloud with Open3D
    pcd = o3d.io.read_point_cloud(fileDIR)
    o3d.visualization.draw_geometries([pcd])
    
    #RANSAC plane fitting
    inlier_cloud, outlier_cloud, normal_vector = plane_fitting(pcd, dist_thresh=0.005, 
                                                ransac_nmb=3, it=1000)
    
    #DBSCAN -> clustering method on outlier point cloud
    pcd_seg, labels = segmentation(outlier_cloud, eps_value=0.08, min_nmb_points=1000)
    
    #Extract object from segmented data
    obj = extract_object(pcd_seg, labels, label_nmb=1)
    
    #Remove outliers
    cl = remove_outliers(obj,nb_neigh=30, std=2)
    
    #Extract bounding boxes
    aabb, obb = get_bounding_box(cl)
    obb.scale(BB_factor,obb.get_center())
    # o3d.visualization.draw_geometries([pcd,aabb,obb])
    o3d.visualization.draw_geometries([pcd,obb])
    
    # draw_template(pcd, "Base-Top_Plate")
    
    #Keep only points inside bounding box
    corners_aabb = np.array(aabb.get_box_points())
    corners_obb = np.array(obb.get_box_points())
    
    inside_rect = inside_rectangle(pcd, corners_obb)
        
    # cl = remove_outliers(inside_rect,nb_neigh=30, std=2)
    # src_mean = np.asarray(inside_rect.get_center())
    # draw_template(pcd, src_mean, normal_vector, 42*math.pi/180, "Base-Top_Plate")
    
    #Denoise point cloud
    #denoised = guided_filter(inside_rect, 0.01, 0.01)
    
    #Find max/min x/y values for finding conversion px <-> meters
    #coef_x, coef_y = m_to_px(pcd, 1280, 720)
    
    #Remove points with depth larger than desired value
    #depth_removed = remove_after_z(pcd, 0.70)
    
    #Remove points outside of px rectangle
    #rect_only = keep_rectangle(pcd, (540,260), (740, 460),coef_x, coef_y, 1280, 720)
    
    normal_pc = estimate_normals(inside_rect, r = 0.1, neigh_max = 30)
    
    #Save result (source) as text file
    # save(inside_rect,name = PC_file + "_BB_" + str(BB_factor),Normals=False)
    # save(normal_pc,name = PC_file + "_BB_" + str(BB_factor),Normals=True)

if __name__ == '__main__':
    main()