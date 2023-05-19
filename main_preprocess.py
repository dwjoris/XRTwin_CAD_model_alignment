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

if __name__ == '__main__':

    """ PROCESS RAW POINT CLOUD """
    # RealSense Raw Point Cloud FileName
    # :: Save files in "h5_files/realsense" directory
    realsense_source_file = "Pendulum_7"
    # Post process raw point cloud and save into .txt file
    post.main(realsense_source_file,BB_factor = 1)
    
    """ COMBINE PROCESSED SOURCE w/ TEMPLATE & GROUND TRUTH """
    # Source .txt FileName
    # :: Processed results saved in "h5_files/output/processing_results" directory
    source_file = "Pendulum_6_BB_1.8_Normals";
    # Template CAD model name
    # :: CAD models saved in "datasets/CAD/" + CAD_Folder_Name + "/" + CAD_name
    CAD_Folder_Name = "Original"
    CAD_name = "Pendulum"
    
    # .hdf5 Result File Name
    result_file_name = source_file
    
    # Ground truth parameters
    Plane_Normal_Vector = [ 0.0074, 0.6649, 0.7469] # Normal vector on table
    X_angle_correction = 0 # Correction for rortation around X-axis (degrees) 
    Y_angle = -45 # rotation around Y-axis (degrees) 
    Z_angle = 0 # rotation around Z-axis (degrees) 
    translation_correction = [0.003,0.050,-0.034] # translation vector correction (meter) 
    mul = 2 # Template Nmb Points = mul * Source Nmb Points 

    # Create .hdf5 file w/ template, source & ground truth
    input_writer.main(source_file, CAD_name, Y_angle, Z_angle, Plane_Normal_Vector,
                        X_angle_correction, translation_correction, multiple = mul,scale=100, 
                        Folder_name= CAD_Folder_Name, result_file = result_file_name,
                        remove_mean=False)
    
    """ ADD SYMMETRIC SOLUTIONS """
    # .hdf5 file directory
    file_loc = "h5_files/output/experiments/" + result_file_name + ".hdf5"
    
    # Add symmetric ground truth solutions
    # :: Ground truth from .hdf5 file is rotated over chosen axis/angle
    add_symm.main(file_loc,180,"z")
    add_symm.main(file_loc,90,"y")
    add_symm.main(file_loc,-90,"y")
    add_symm.main(file_loc,180,"y")
    add_symm.main(file_loc,180,"x")
    add_symm.main(file_loc,180,"x",90,"y")
    add_symm.main(file_loc,180,"x",-90,"y")
    add_symm.main(file_loc,180,"x",180,"y")
    
    """ COMBINE h5 FILES INTO 1 (unused)"""
    # #h5 Files directory
    # DIR = "C:/Users/menth/Documents/Python Scripts/Thesis/h5_files/output/experiments/Old"
    # # Result filename
    # File_Name = "No_Normals_Test"
    # # Combine h5 files from given directory into 1 file
    # h5_combine.main(DIR, File_Name)
    
    