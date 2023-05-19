"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



main errors

Compute errors for all result files in given directory

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth
        o Estimated transformation

Output:
    - .xlsx file with errors
    - .xlsx file with variances
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

# misc imports
import misc.compute_errors as compute_errors
from misc.xlsx_handler import save_errors_to_xlsx as save_errors_excel
# import misc.visualise_results as visualiser
from misc.errors import Errors
from misc.plotter import failure_division_plot


def auto_error_computation(DIR, zero_mean = True, nmb_it = 1, MRAE_lim = 120, R2_lim = 0, Recall_lim = 0.01):
    # Initalize zero error list to compute mean
    error_class = Errors(0,0,0,0)

    # Loop over all objects in results folder
    for i,object_name in enumerate(tqdm(os.listdir(DIR))):
        Object_Folder_Path = os.path.join(DIR, object_name)
        
        # Save errors/failure cases per object
        failure_cases = 0
        scan_nmb = -1;
        
        errors_tot = np.zeros((9,1))
        nmb_files = len([name for name in os.listdir(Object_Folder_Path)])
        nmb_scans = int(nmb_files/nmb_it)
        # List of number of failures/scan
        failure_div_list = np.zeros(nmb_scans)
        
        # Loop over all scans for 1 object
        for j,scan_name in enumerate(os.listdir(Object_Folder_Path)):
            # Increase scan_nmb if results are for new scan
            if(j%nmb_it == 0):
                scan_nmb = scan_nmb + 1
                
            # Check if current path is a file   
            if os.path.isfile(os.path.join(Object_Folder_Path, scan_name)):
                
                # Current file path
                file_loc = os.path.join(Object_Folder_Path, scan_name)
                           
                # visualiser.main(file_loc, zero_mean=False)
                errors = compute_errors.main(file_loc,Recall_lim,symm_sol=True,zero_mean=zero_mean)
                # print(errors)
                
                if(errors[7,0] < R2_lim or abs(errors[0,0]) > MRAE_lim):
                    failure_cases = failure_cases + 1
                    failure_div_list[scan_nmb] = failure_div_list[scan_nmb] + 1
                    # visualiser.main(file_loc,zero_mean=True)
                else:
                    # errors_tot = errors_tot + errors
                    errors_tot = np.append(errors_tot,errors,1)
                    # visualiser.main(file_loc,zero_mean=True)
        
        # Compute mean & variances per object, remove first column since only zeroes
        mean, variance = compute_errors.compute_mean_variance(errors_tot[:,1:])
        # print("\n:: Mean results " + object_name)   
        # error_class.display(np.expand_dims(mean,1))
        # print("\n:: Variance results " + object_name)   
        # error_class.display(np.expand_dims(variance,1))
        save_errors_excel(np.expand_dims(mean,1), np.expand_dims(variance,1), object_name)
        # print("\n:: Number of failure cases: ", failure_cases)
        failure_division_plot(failure_div_list/nmb_files*100, nmb_scans, object_name)

if __name__ == '__main__':

    """ VARIABLES """
    # Failure cases when R2 < R2_lim and MRAE > MRAE_lim
    R2_lim   = 0;
    MRAE_lim = 120;
    
    # Directory of files (containing folders of results per object)
    BASE_DIR = "C:/Users/menth/Documents/Python Scripts/Thesis/h5_files/output/results/"
    DIR      = BASE_DIR + "ICP/GO-ICP_Ref/MSE_0.0001_trim_0.0001_ICP_VS_0.1"
    
    # Number of iterations from testing
    nmb_it = 2
    
    """ COMPUTE ERRORS """
    auto_error_computation(DIR, zero_mean = False, nmb_it = nmb_it)
