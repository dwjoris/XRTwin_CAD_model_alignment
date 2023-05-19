"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



main optimize

Run selected registration method, for all scans, for series of voxel sizes

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth

Output:
    - array file with errors
"""


"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

# General
import os
from tqdm import tqdm
import numpy as np
import shutil
import matplotlib.pyplot as plt

# misc imports
import misc.compute_errors as compute_errors
from misc.xlsx_handler import save_errors_to_xlsx as save_errors_excel

# _test imports (uncomment desired method)
# import _test.test_PointNetLK_OwnData as PNLK_test
# import _test.test_PointNet_OwnData as PN_test
# import _test.test_RPMNet_OwnData as RPMN_test
# import _test.test_ROPNet_OwnData as ROPN_test
# import _test.test_FGR_OwnData as FGR_test
import _test.test_RANSAC_OwnData as RANSAC_test
# import _test.test_GOICP_OwnData as GOICP_test
# import _test.test_ICP_OwnData as ICP_test
# import _test.test_PRNet_OwnData as PRNet_test


"""
=============================================================================
---------------------------------FUNCTIONS-----------------------------------
=============================================================================
"""


def auto_error_computation(DIR, zero_mean = True, nmb_it = 1, MRAE_lim = 120, R2_lim = 0, recall_lim = 0.01):
    # Function saves errors/failure cases per object
    
    # Initialize variables
    failure_cases = 0
    scan_nmb = -1;
    errors_tot = np.zeros((9,1))
    
    # Loop over results in directory    
    for j,scan_name in enumerate(os.listdir(DIR)):
        # Increase scan_nmb if results are for new scan
        if(j%nmb_it == 0):
            scan_nmb = scan_nmb + 1
                
        # Check if current path is a file   
        if os.path.isfile(os.path.join(DIR, scan_name)):
                
            # Current file path
            file_loc = os.path.join(DIR, scan_name)
            # print(file_loc)
                           
            errors = compute_errors.main(file_loc,recall_lim,symm_sol=True,zero_mean=zero_mean)
                
            if(errors[7,0] < R2_lim or abs(errors[0,0]) > MRAE_lim):
                failure_cases = failure_cases + 1
            else:
                errors_tot = np.append(errors_tot,errors,1)
        
    # Compute mean & variances per object, remove first column since only zeroes
    mean, variance = compute_errors.compute_mean_variance(errors_tot[:,1:])
    save_errors_excel(np.expand_dims(mean,1), np.expand_dims(variance,1), scan_name)
    return np.expand_dims(mean,1), failure_cases

def apply_registration(exp_name, DIR, nmb_it = 1, zero_mean = True, voxel_size = 0, refine = True):
    # Function loops over all files in given directory and saves results for selected method
    for j,scan_name in enumerate(os.listdir(DIR)):
    # Check if current path is a file
        if os.path.isfile(os.path.join(DIR, scan_name)):
                
            # Current file path
            file_loc = os.path.join(DIR, scan_name)
                
            for it in range(nmb_it):
                name = exp_name
                # registration_time = RPMN_test.main(file_loc, name,zero_mean=False,voxel_size=0.005)
                # registration_time = FGR_test.main(file_loc,object_name,voxel_size=voxel_size,zero_mean=True)
                # registration_time = GOICP_test.main(file_loc,object_name,MSEThresh=0.0001,trimFraction=0.0001)
                # registration_time = ICP_test.main(file_loc, name, refine = refine,voxel_size=voxel_size,zero_mean=zero_mean)
                RANSAC_test.main(file_loc, name,voxel_size=voxel_size,zero_mean=zero_mean)                
                # registration_time = PNLK_test.main(file_loc,object_name,zero_mean=False,voxel_size=voxel_size)
                # registration_time = PRNet_test.main(file_loc,object_name)
                # registration_time = ROPN_test.main(file_loc,name,zero_mean=False,voxel_size=voxel_size)

"""
=============================================================================
---------------------------------VARIABLES-----------------------------------
=============================================================================
"""

""" Choose experiment name """

exp_name = "experiment_name"

""" Base/Input/Output directory of results/data """

BASE_DIR = "C:/Users/menth/Documents/Python Scripts/Thesis/h5_files/output/"
# Directory with input files
IN_DIR  = BASE_DIR + "experiments/Realsense_BB_1/Base-Top_Plate" 
# Directory where results are saved 
OUT_DIR = BASE_DIR + "results/ICP/" + exp_name

""" Registration parameters """

nmb_it = 20          # Number of iterations per scan
zero_mean = False    # Center on origin (True: Yes, False: No)

# Range of voxel sizes to test    
vs_low = 0.005; vs_high = vs_low*2; vs_step = vs_low
vs_range = np.arange(vs_low, vs_high, vs_step)

# To save results
mean_errors_tot = np.zeros((9,1))
failure_cases_tot = np.zeros(1)

# Loop over voxel sizes
for voxel_size in vs_range:
        
   # Apply registration
   apply_registration(exp_name, IN_DIR, nmb_it, zero_mean, voxel_size, refine = True)
   # Compute errors
   mean_errors, failure_cases = auto_error_computation(OUT_DIR,zero_mean,nmb_it)
   print(failure_cases)
   mean_errors_tot   = np.append(mean_errors_tot,mean_errors,1)
   failure_cases_tot = np.append(failure_cases_tot,failure_cases)
   
   # Delete previous results
   try:
       shutil.rmtree(OUT_DIR)
   except OSError as e:
       print("Error: %s - %s." % (e.filename, e.strerror))
            
# Save all errors and number of failure cases to a numpy file
np.save(OUT_DIR + '/nmb_failures',failure_cases_tot)
np.save(OUT_DIR + '/errors',mean_errors_tot)

# Plot evolution of failure cases as a function of the voxel size
# plt.plot(np.arange(0,0.1,0.0005),failure_cases_tot)
                

