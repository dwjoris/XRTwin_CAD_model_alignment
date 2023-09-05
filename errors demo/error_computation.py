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
    - numpy array of individual errors
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
import functions.compute_errors as compute_errors
from functions.misc import save_errors_to_xlsx as save_errors_excel
from functions.misc import failure_division_plot
# import misc.visualise_results as visualiser
from functions.errors import Errors

"""
=============================================================================
---------------------------------FUNCTIONS-----------------------------------
=============================================================================
"""

def error_criteria(R2_value, MRAE_value, R2_lim, MRAE_lim):
    """
    Define criteria to consider case a failure
    
    Parameters
    ----------
    R2_value            : float     // Computed value for R2 metric
    MRAE_value          : float     // Computed value for MRAE metric
    R2_lim              : float     // R2 for identifying failure case
    MRAE_lim            : float     // MRAE for identifying failure case
    
    Returns
    ----------
    boolean to signal failure/not
    """
    
    if(R2_value < R2_lim or abs(MRAE_value) > MRAE_lim):
        return True
    else:
        return False


def auto_error_computation(DIR, zero_mean = True, nmb_it = 1, MRAE_lim = 120, R2_lim = 0, Recall_lim = 0.01):
    """
    Compute errors for all results, per object
    
    Parameters
    ----------
    DIR                 : String    // To results folder
    zero_mean           : Boolean   // To signal whether point clouds are centered
    nmb_it              : int       // number of testing runs per scan
    MRAE_lim            : float     // MRAE for identifying failure case
    R2_lim              : float     // R2 for identifying failure case
    Recall_lim          : float     // Limit for recall error
    
    Returns
    ----------
    .xlsx file with computed mean errors
    numpy array containing individual errors
    """
    
    # Initalize zero error list to compute mean
    error_class = Errors(0,0,0,0)
    
    # Save all errors in array
    errors_tot = np.zeros((8,1))

    # Loop over all objects in results folder
    for object_index, object_name in enumerate(tqdm(os.listdir(DIR))):
        Object_Folder_Path = os.path.join(DIR, object_name)
        
        # Save errors/failure cases per object
        failure_cases = 0
        scan_nmb = -1;
        
        # Save only errors when not failure case
        errors_tot_lim = np.zeros((8,1))
        
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
                errors_tot = np.append(errors_tot,errors,1)
                
                if(error_criteria(errors[7,0], errors[0,0], R2_lim, MRAE_lim)):
                    failure_cases = failure_cases + 1
                    failure_div_list[scan_nmb] = failure_div_list[scan_nmb] + 1
                    # visualiser.main(file_loc,zero_mean=True)
                else:
                    errors_tot_lim = np.append(errors_tot_lim,errors,1)
                    # visualiser.main(file_loc,zero_mean=True)
        
        # Compute mean & variances per object, remove first column since only zeroes
        mean, variance = compute_errors.compute_mean_variance(errors_tot_lim[:,1:])
        print("\n:: Mean results " + object_name)   
        error_class.display(np.expand_dims(mean,1))
        print("\n:: Variance results " + object_name)   
        error_class.display(np.expand_dims(variance,1))
        save_errors_excel(np.expand_dims(mean,1), np.expand_dims(variance,1), object_name)
        print("\n:: Number of failure cases: ", failure_cases)
        failure_division_plot(failure_div_list/nmb_files*100, nmb_scans, object_name)

    return errors_tot[:,1:]

"""
=============================================================================
-------------------------------EXECUTE CODE----------------------------------
=============================================================================
"""

""" VARIABLES """
# Failure cases when R2 < R2_lim and MRAE > MRAE_lim
R2_lim   = 0; 
MRAE_lim = 120;

# Recall limit for recal error definition (# points for which RMSE < recall lim)
Recall_lim = 0.01
    
# Directory of files (containing folders of results per object)
BASE_DIR = os.getcwd()
DIR      = BASE_DIR + "\\results"
    
# Number of iterations from testing
nmb_it = 2

# Point clouds centered on origin
zero_mean = False;
    
""" COMPUTE ERRORS """
errors_tot = auto_error_computation(DIR, zero_mean = zero_mean, nmb_it = nmb_it, MRAE_lim = MRAE_lim, 
                                    R2_lim = R2_lim,Recall_lim = Recall_lim)
