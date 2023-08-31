import open3d as o3d
import numpy as np
import math
import torch

def prepare_stl(template_stl, nmb_source_points, multiple, 
                Normals_radius = 0.01, Normals_Neighbours = 30):
    """
    Prepare .stl file to .ply file
    
    Parameters
    ----------
    template_stl : Open3d TriangleMesh Object
    nmb_source_points : int
    multiple : float
    Normals_radius : float (to estimate normal vectors on template)
    Normals_Neighbours : float (to estimate normal vectors on template)
    """
    
    nmb_template_points = nmb_source_points*multiple
    template_pointcloud = template_stl.sample_points_uniformly(number_of_points=nmb_template_points)  

    template_pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=Normals_radius, max_nn=Normals_Neighbours))

    return template_pointcloud

def create_rotation_matrix(angle,axis):
    """
    Create rotation matrix from given angle and axis

    Parameters
    ----------
    angle : float (defrees)
    axis : string (x/y/z)
       
    Returns
    -------
    rotation_matrix : 3x3 Numpy array
    """
    
    angle = angle*math.pi/180
    R = np.eye(3)
    if(axis == "z"): # This sign is wrong
        R[0,:] = [math.cos(angle), math.sin(angle), 0]
        R[1,:] = [-math.sin(angle), math.cos(angle), 0]
    elif(axis == "y"):
        R[0,:] = [math.cos(angle), 0, math.sin(angle)]
        R[2,:] = [-math.sin(angle),0, math.cos(angle)]
    else:
        R[1,:] = [0, math.cos(angle), -math.sin(angle)]
        R[2,:] = [0, math.sin(angle), math.cos(angle)]
    return R

def get_rotation(Plane_Normal_Vector, x_angle_correction, y_angle, z_angle):
    """
    Get rotation matrix from Menthy's original groundtruth data

    Parameters
    ----------
    Plane_Normal_Vector : 3x1 numpy array
    x_angle_correction : float
    y_angle : float
    z_angle : float
       

    Returns
    -------
    rotation_matrix : 3x3 Numpy array
    """
    
    # Angles
    Plane_Normal_Norm = np.linalg.norm(np.array(Plane_Normal_Vector))

    x_angle = math.acos(Plane_Normal_Vector[1]/Plane_Normal_Norm)*180/math.pi
    # print("Computed angle over x-axis is %.3f.\n" % x_angle)
    x_angle = x_angle + x_angle_correction
    
    # Rotation matrix,    Ps = R*Pt
    Rz = create_rotation_matrix(z_angle,"z")
    Rx = create_rotation_matrix(x_angle,"x")
    Ry = create_rotation_matrix(y_angle,"y")
    
    # Multiply rotation matrices on rigth for rotation over new axis
    Rtemp = np.matmul(Rx,Rz)
    rotation_matrix = np.matmul(Rtemp,Ry)
    return rotation_matrix


def prepare_ply(point_cloud, scale):
    """
    Transforms point cloud into tensor 
    
    Parameters
    ----------
    point_cloud : Open3d PointCloud Object
    
    return tensor
    """
    
    array = np.asarray(point_cloud.points)/scale
    DIM = 3
    
    if(point_cloud.has_normals()):
        normals_array = np.asarray(point_cloud.normals)
        array = np.concatenate((array,normals_array),1)
        DIM = 6
    
    tensor = torch.from_numpy(array)                                                                                                                            
    tensor = tensor.expand(1,tensor.size(0),DIM)
    
    tensor_mean = torch.mean(tensor, 1)                                           
    tensor = tensor - tensor_mean   
    return tensor
    
# def homogenous_transfo(rotation_matrix,translation_vector): 
#     """
#     Creates 4x4 transformation matrix 
    
#     Parameters
#     ----------
#     rotation_matrix : 3x3 numpy array
#     translation_vector: 3x1 list
    
#     return T
#     """
    
#     T = np.zeros((4,4))
#     T[3,3] = 1
#     T[0:3,0:3] = rotation_matrix
#     T[:3,3] = translation_vector
#     return T


def apply_transfo(tensor,R,t,normals=True):
    """
    Applies given rotation and translation on tensor
    
    Parameters
    ----------
    tensor : 1xNx6 torch tensor
    R: 3x3 numpy array
    t: 3x1 numpy array
    normals: boolean
    
    return transformed tensor
    """
    
    if(normals == False):
        transformed_tensor = np.zeros((1,tensor.shape[1],3))
        for i in range(tensor.shape[1]):
            transformed_tensor[0][i][:] = np.add(np.matmul(R,tensor[0][i][:]),t)
    else:
        transformed_tensor = np.zeros((1,tensor.shape[1],6))
        for i in range(tensor.shape[1]):
            transformed_tensor[0][i][0:3] = np.add(np.matmul(R,tensor[0][i][0:3]),t)
            transformed_tensor[0][i][3:6] = np.matmul(R,tensor[0][i][3:6])  # Normals only rotated
    return transformed_tensor

def ground_truth_estimation(translation_correction, rotation_matrix, source_pointcloud, 
                            template_pointcloud):
    """
    Full Ground Truth computation
    
    Parameters
    ----------
    Plane_Normal_Vector : 3x1 array
    x_angle_correction : float
    y_angle : float
    z_angle : float
    source_pointcloud : Open3D pointcloud
    template_pointcloud : Open3D pointcloud, averaged from the CAD model
    """
    
    # Turn .ply to tensor and remove mean
    source_tensor = prepare_ply(source_pointcloud, scale = 1)
    template_tensor = prepare_ply(template_pointcloud, scale = 100)
    
    # Turn translation vector to numpy array
    translation_tensor = np.array(translation_correction)
    
    # Transform template with ground truth
    template_tensor_transformed = apply_transfo(template_tensor,rotation_matrix,translation_tensor)
    
    # Show results in Open3d
    show_open3d(template_tensor_transformed[:,:,0:3], source_tensor[:,:,0:3])
    
    return

def show_open3d(template,source, name = "Open3D"):
    """
    Show both template and source pointclouds
    
    Parameters
    ----------
    template : 1xNx6 tensor
    source: 1xMx6 tensor
    name : string
    """
    
    # Create point clouds
    template_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template[0][:][:])
    source_.points = o3d.utility.Vector3dVector(source[0][:][:])
    
    # Add color
    template_.paint_uniform_color([0,0,1])
    source_.paint_uniform_color([1,0,0])
    
    o3d.visualization.draw_geometries([template_,source_],window_name=name)
    
"""
-----------

EXECUTE CODE 

-----------

"""   
    
"""
Define Variables
---------------
Read file names
"""

Source_PLY_File = "Base-Top_Plate_1_Source.ply"
Template_STL_File = "Base-Top_Plate.stl"

# Ground truth parameters
Plane_Normal_Vector = [ 0.0038, 0.6526, 0.7577] # First estimate of rotation around X-axis
X_angle_correction = 0 # Correction for rortation around X-axis (degrees) 
Y_angle = 90 # rortation around Y-axis (degrees) 
Z_angle = 0
translation_correction = [0.005,0.008,-0.02]
mul = 2

"""
Initializations
---------------
Turn files into Open3D PointCloud Object
"""

source_pointcloud = o3d.io.read_point_cloud(Source_PLY_File)
template_stl = o3d.io.read_triangle_mesh(Template_STL_File)  

nmb_source_points = len(np.asarray(source_pointcloud.points))
template_pointcloud = prepare_stl(template_stl, nmb_source_points, mul)

"""
Ground Truth Estimation
---------------
Estimate ground truth transformation, applied on template --> source
"""

rotation_matrix = get_rotation(Plane_Normal_Vector, X_angle_correction, Y_angle, Z_angle)

ground_truth_estimation(translation_correction, rotation_matrix, source_pointcloud, template_pointcloud)