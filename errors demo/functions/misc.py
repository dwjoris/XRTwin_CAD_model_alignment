
"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

import os
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows
import matplotlib.pyplot as plt

"""
=============================================================================
--------------------------------DATA SAVING----------------------------------
=============================================================================
"""

def save_dataframe_to_xlsx(array, index_name, column_names, result_file, sheet_name):
    """
    Saves given array to .xlsx file
    
    Parameters
    ----------
    array               : RxC numpy array               // data to save in .xlsx file 
    index_name          : Rx1 list of Strings           // name of rows
    column_names        : 1xC list of Strings           // name of columns
    result_file         : String                        // result file name
    sheet_name          : String                        // name of sheet
    """
    
    """
    Source: https://stackoverflow.com/a/69258865
    """
    
    # Create dataframe from array
    df = pd.DataFrame(array,
                      index=[index_name],
                      columns=column_names)
    
    # Check if .xlsx file already exists
    if not os.path.exists(result_file):
        df.to_excel(result_file, sheet_name = sheet_name)
    else:
        workbook = openpyxl.load_workbook(result_file)  # load workbook if already exists
        sheet = workbook[sheet_name]  # declare the active sheet 
        # append the dataframe results to the current excel file
        for row in dataframe_to_rows(df, header = False, index=[index_name]):
            sheet.append(row)
        workbook.save(result_file)  # save workbook
        workbook.close()  # close workbook

def save_errors_to_xlsx(mean_array, variance_array, object_name, result_file = "saved_results"):
    """
    Saves array of mean errors and variances to .xlsx file with desired name
    
    Parameters
    ----------
    mean_array              : 8x1 numpy array               // mean data 
    variance_array          : 8x1 numpy array               // variance data
    object_name             : String                        // name of object 
    result_file             : String                        // result file name
    """
    
    # Transpose arrays
    mean_array_tp = np.transpose(mean_array)
    var_array_tp  = np.transpose(variance_array)
    
    # Column names (error names)
    column_names = ['RE (°)','RE (mm)','RMSE (°)','RMSE (mm)','MAE (°)','MAE (mm)','Recall','R2']
    
    # Save to .xlsx
    save_dataframe_to_xlsx(mean_array_tp, object_name, column_names, result_file + '_mean.xlsx', 'mean_results')
    save_dataframe_to_xlsx(var_array_tp, object_name, column_names, result_file + '_var.xlsx', 'variance_results')
    
"""
=============================================================================
------------------------------DATA PLOTTING----------------------------------
=============================================================================
"""

def failure_division_plot(failure_div_list, nmb_scans, object_name):
    """
    Used to plot amount of failures for each scan
    
    Parameters
    ----------
    failure_div_list            : Mx0 numpy array               // number of failure cases per scan
    nmb_scans                   : int                           // number of scans for object
    object_name                 : String                        // name of object 
    """
    
    scan_names = range(1,nmb_scans+1)
    
    fig1, ax1 = plt.subplots()
    ax1.bar(scan_names, failure_div_list)
    ax1.set_title("Failure division among scans for " + object_name)
    ax1.set_ylabel("Number of failures (%)")
    ax1.set_xlabel("Scan Number")
    ax1.legend()