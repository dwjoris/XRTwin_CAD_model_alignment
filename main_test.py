"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



main test

Run selected registration method, for all objects, all scans, for given parameters

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth

Output:
    - array file with computation times
    - .hdf5 files with estimated transformation

Credits: 
    PointNetLK, RPMNet, ROPNet & PRNet Code by vinits5 as part of the Learning3D library 
    Link: https://github.com/vinits5/learning3d#use-your-own-data
    
    RANSAC, FGR as part of the Open3D library
    
    GO-ICP Code by aalavandhaann
    Link: https://github.com/aalavandhaann/go-icp_cython

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""
# General
import os
from tqdm import tqdm

# misc imports
from misc.xlsx_handler import save_timing_to_xlsx as save_time_excel

# _test imports
# import _test.test_PointNetLK_OwnData as PNLK_test
# import _test.test_PointNet_OwnData as PN_test
# import _test.test_RPMNet_OwnData as RPMN_test
# import _test.test_ROPNet_OwnData as ROPN_test
# import _test.test_FGR_OwnData as FGR_test
# import _test.test_RANSAC_OwnData as RANSAC_test
# import _test.test_GOICP_OwnData as GOICP_test
import _test.test_ICP_OwnData as ICP_test
# import _test.test_PRNet_OwnData as PRNet_test

"""
=============================================================================
----------------------------------FUNCTIONS----------------------------------
=============================================================================
"""

def assign_voxelsize(object_name):
    if(object_name == "Base-Top_Plate"):
        return 0.005
    elif(object_name == "Pendulum"):
        return 0.005
    elif(object_name == "Separator"):
        return 0.005
    elif(object_name == "Round-Peg"):
        return 0.001
    elif(object_name == "Square-Peg"):
        return 0.001
    elif(object_name == "Shaft-New"):
        return 0.001
    elif(object_name == "Range-Hood"):
        return 0.005
    elif(object_name == "Guitar"):
        return 0.008
    
    return 0.005

"""
=============================================================================
------------------------------------MAIN------------------------------------
=============================================================================
"""

if __name__ == '__main__':
    
    """ REGISTRATION TEST """
    
    # Name of experiment (folder name of results)
    exp_name = "GO-ICP_Ref/MSE_0.1_trim_0.0001_ICP_VS_0.1"
    # Directory of input .hdf5 files
    # :: Folder containing subfolders for each object w/ input files
    DIR = "C:/Users/menth/Documents/Python Scripts/Thesis/h5_files/output/results/GO-ICP/MSE_0.1_trim_0.0001"
    # Number of runs per scan
    nmb_it = 1
    
    # To save registration times
    reg_times = []
    
    # Run over all objects in input folder
    for i,object_name in enumerate(tqdm(os.listdir(DIR))):
        Object_Folder_Path = os.path.join(DIR, object_name)
        registration_time_tot = 0
        count = 0
        
        # Run over all scans for object
        for j,scan_name in enumerate(tqdm(os.listdir(Object_Folder_Path))):
            # Check if current path is a file
            if os.path.isfile(os.path.join(Object_Folder_Path, scan_name)):
                
                # Current file path
                file_loc = os.path.join(Object_Folder_Path, scan_name)
                
                for it in range(nmb_it):
                    # Assign voxel size per object
                    # voxel_size = assign_voxelsize(object_name)
                    name = exp_name + "/" + object_name
                    
                    # Choose registration method w/ parameters
                    # registration_time = RPMN_test.main(file_loc, name,zero_mean=False,voxel_size=0.005)
                    # registration_time = FGR_test.main(file_loc,object_name,voxel_size=voxel_size,zero_mean=True)
                    # registration_time = GOICP_test.main(file_loc,object_name,MSEThresh=0.0001,trimFraction=0.0001)
                    registration_time = ICP_test.main(file_loc, name, refine = True,voxel_size=0.1,zero_mean=False)
                    # registration_time = RANSAC_test.main(file_loc, object_name,voxel_size=0.1,zero_mean=True)                
                    # registration_time = PNLK_test.main(file_loc,object_name,zero_mean=False,voxel_size=voxel_size)
                    # registration_time = PRNet_test.main(file_loc,object_name)
                    # registration_time = ROPN_test.main(file_loc,name,zero_mean=False,voxel_size=voxel_size)
                    
                    registration_time_tot = registration_time_tot + registration_time
                    count = count + 1
        reg_times.append([object_name, registration_time_tot/count])
        # print("\n:: Average registration time: ", registration_time_tot/count)
    # Save registration times to .xlsx file
    save_time_excel(reg_times,"ICP_Ref")

    
   