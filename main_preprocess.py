"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



main preprocess

Pre-proces obtained point clouds

Inputs:
    - .ply file (Source, RealSense)

Output:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth (w/ symmetric solutions)
        o Estimated transformation
"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

# h5_files imports
import h5_files.input_writer as input_writer
import h5_files.add_symmetric as add_symm
# import h5_files.h5_combine as h5_combine
import h5_files.pcl_post_processing as post
import h5_files.h5_writer as w
import math
import numpy as np


def get_rotation(Plane_Normal_Vector, x_angle_correction, y_angle, z_angle):
    """
    Get rotation matrix from Menthy's original groundtruth data

    Parameters
    ----------
    Plane_Normal_Vector : 3x1 array
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
    Rz = w.rotation_matrix(z_angle,"z")
    Rx = w.rotation_matrix(x_angle,"x")
    Ry = w.rotation_matrix(y_angle,"y")
    
    # Multiply rotation matrices on rigth for rotation over new axis
    Rtemp = np.matmul(Rx,Rz)
    rotation_matrix = np.matmul(Rtemp,Ry)
    return rotation_matrix


if __name__ == '__main__':

    """ PROCESS RAW POINT CLOUD """
    # RealSense Raw Point Cloud FileName
    # :: Save files in "h5_files/realsense" directory
    # realsense_source_file = "Base_Top_Plate_1_FP"
    # Post process raw point cloud and save into .txt file
    # post.main(realsense_source_file,BB_factor = 1)
    
    """ COMBINE PROCESSED SOURCE w/ TEMPLATE & GROUND TRUTH """
    # Source .txt FileName
    # :: Processed results saved in "h5_files/output/processing_results" directory
    source_file = "Base-Top_Plate_1_BB_1_Normals";
    # Template CAD model name
    # :: CAD models saved in "datasets/CAD/" + CAD_Folder_Name + "/" + CAD_name
    CAD_Folder_Name = "Original"
    CAD_name = "Base-Top_Plate"
    
    # .hdf5 Result File Name
    result_file_name = source_file
    
    # Ground truth parameters
    Plane_Normal_Vector = [ 0.0038, 0.6526, 0.7577] # First estimate of rotation around X-axis
    X_angle_correction = 0 # Correction for rortation around X-axis (degrees) 
    Y_angle = 90 # rortation around Y-axis (degrees) 
    Z_angle = 0
    translation_correction = [0.005,0.008,-0.02]
    mul = 2
    
    # Rotation Matrix
    rotation_matrix = get_rotation(Plane_Normal_Vector, X_angle_correction, Y_angle, Z_angle)
    
    # Create .hdf5 file w/ template, source & ground truth
    input_writer.main(source_file, CAD_name, Y_angle, Z_angle, Plane_Normal_Vector,
                        X_angle_correction, translation_correction, multiple = mul,scale=100, 
                        Folder_name= CAD_Folder_Name, result_file = result_file_name,
                        remove_mean=False)
    
    """ ADD SYMMETRIC SOLUTIONS """
    # .hdf5 file directory
    # file_loc = "h5_files/output/experiments/" + result_file_name + ".hdf5"
    
    # Add symmetric ground truth solutions
    # :: Ground truth from .hdf5 file is rotated over chosen axis/angle
    # add_symm.main(file_loc,180,"z")
    # add_symm.main(file_loc,90,"y")
    # add_symm.main(file_loc,-90,"y")
    # add_symm.main(file_loc,180,"y")
    # add_symm.main(file_loc,180,"x")
    # add_symm.main(file_loc,180,"x",90,"y")
    # add_symm.main(file_loc,180,"x",-90,"y")
    # add_symm.main(file_loc,180,"x",180,"y")
    
    """ COMBINE h5 FILES INTO 1 (unused)"""
    # #h5 Files directory
    # DIR = "C:/Users/menth/Documents/Python Scripts/Thesis/h5_files/output/experiments/Old"
    # # Result filename
    # File_Name = "No_Normals_Test"
    # # Combine h5 files from given directory into 1 file
    # h5_combine.main(DIR, File_Name)
    
    