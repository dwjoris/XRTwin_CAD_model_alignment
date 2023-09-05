"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



compute errors

Compute the different metrics based on the template, source, transformed source,
ground truth & estimated transformation.

Inputs:
    - .hdf5 files containing:
        o Template (w/ normals)
        o Source   (w/ normals)
        o Ground Truth
        o Estimated Transformation

Output:
    - Relative Errors
    - Root Mean Square Errors
    - Mean Absolute Errors
    - Recall
    - R2 
"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import time
import numpy as np
from functions.errors import Errors
from functions.dataloader import h5file_to_torch

"""
=============================================================================
------------------------------------CODE-------------------------------------
=============================================================================
"""

def compute_mean_variance(error_list):
    """
    Compute mean and variance for given list of errors
    
    Parameters
    ----------
    error_list          : 8xN numpy array   // List of computed errors
    
    Returns
    ----------
    mean                : 8x1 numpy array   // Means for every error
    variance            : 8x1 numpy array   // Variance for every error             
    """
    
    mean = np.mean(error_list,1)
    variance = np.var(error_list,1)
    return mean, variance

def symmetric_errors(templ_tensor, src_tensor, gt_symm_tensor, transfo_tensor, recall_lim):
    """
    Compute smallest error in the case of symmetric solutions
    
    Parameters
    ----------
    templ_tensor        : Nx6 torch tensor      // Tensor of template points
    src_tensor          : Nx6 torch tensor      // Tensor of source points
    gt_symm_tensor      : Mx4x4 torch tensor    // Tensor of all ground truth solutions
    transfo_tensor      : Nx6 torch tensor      // Tensor of transformed template points
    recall_lim          : float                 // Limit for recall metric
    
    Returns
    ----------
    errors_min          : 8x1 numpy array       // List of best errors (closest to symmetric solution)
    mean_error_time     : float                 // Time to compute errors         
    """
    
    # Go over all possible ground truth solutions and find best one
    # Initialize error with max possible value (180°)
    max_recall = 0
    nmb_sol = gt_symm_tensor.shape[0]
    error_time = 0
    errors_min = np.ones((8,1))*-1
    # print(nmb_sol)
    
    for i in range(nmb_sol):
        # Compute errors (+ time)
        # Create error module
        
        gt_sol = gt_symm_tensor[i,:,:].expand(1,4,4)
        error_class = Errors(templ_tensor[:,:,0:3],src_tensor[:,:,0:3],gt_sol,transfo_tensor,recall_lim)
        
        # Compute errors
        start = time.time()
        errors = error_class()
        error_time = time.time()-start + error_time
        
        # Display errors
        # error_class.display(errors)
        
        # Save errors if smallest one
        if(errors[6,0] > max_recall):
            errors_min = errors
            max_recall = errors[6,0]
            
    mean_error_time = error_time/nmb_sol
    # Display errors
    # error_class.display(errors_min)
    return errors_min, mean_error_time

def non_symmetric_errors(templ_tensor, src_tensor, gt_tensor, transfo_tensor, recall_lim):
    """
    Compute smallest error in the case of one ground truth solution
    
    Parameters
    ----------
    templ_tensor        : Nx6 torch tensor      // Tensor of template points
    src_tensor          : Nx6 torch tensor      // Tensor of source points
    gt_tensor           : Mx4x4 torch tensor    // Tensor of ground truth solution
    transfo_tensor      : Nx6 torch tensor      // Tensor of transformed template points
    recall_lim          : float                 // Limit for recall metric
    
    Returns
    ----------
    errors              : 8x1 numpy array       // List of errors
    mean_error_time     : float                 // Time to compute errors         
    """
    
    error_class = Errors(templ_tensor[:,:,0:3],src_tensor[:,:,0:3],gt_tensor,transfo_tensor,recall_lim)
    
    # Compute errors
    start = time.time()
    errors = error_class()
    error_time = time.time()-start
    
    # Display errors
    # error_class.display(errors)

    return errors, error_time   

def main(h5_file_loc,recall_lim,symm_sol = False, zero_mean = False): 
    """
    Compute all errors and find best error in case of symmetric solutions
    
    Parameters
    ----------
    h5_file_loc         : String                // location of result .hdf5 file
    recall_lim          : float                 // Limit for recall metric
    symm_sol            : Boolean               // Whether multiple symmetric solutions exist
    zero_mean           : Boolean               // Whether the point clouds were centered during registration
    
    Returns
    ----------
    errors_min          : 8x1 numpy array       // List of (best) errors (closest to symmetric solution)       
    """
    
    # Load data
    templ_tensor, src_tensor, gt_tensor, gt_symm_tensor, transfo_tensor = h5file_to_torch(h5_file_loc, 
                                                                                          zero_mean,
                                                                                          T_est = True)
    
    # In case multiple solutions exist
    if(symm_sol):
        errors_min, mean_error_time = symmetric_errors(templ_tensor, src_tensor, gt_symm_tensor, 
                                                       transfo_tensor, recall_lim)
    else:
        errors_min, mean_error_time = non_symmetric_errors(templ_tensor, src_tensor, gt_tensor, 
                                                       transfo_tensor, recall_lim)
    
    # print(":: Mean error time is: ", mean_error_time)
    
    return errors_min
    
if __name__ == '__main__':
    main()
