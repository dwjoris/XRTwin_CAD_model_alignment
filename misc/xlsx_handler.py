import os
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows


# Credits: https://stackoverflow.com/a/69258865

def save_dataframe_to_xlsx(array, index_name, column_names, result_file, sheet_name):
    df = pd.DataFrame(array,
                      index=[index_name],
                      columns=column_names)
    # print(df)
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
    mean_array_tp = np.transpose(mean_array)
    var_array_tp  = np.transpose(variance_array)
    
    column_names = ['RE (°)','RE (mm)','RMSE (°)','RMSE (mm)','MAE (°)','MAE (mm)','Recall','R2','CD']
    
    save_dataframe_to_xlsx(mean_array_tp, object_name, column_names, result_file + '_mean.xlsx', 'mean_results')
    save_dataframe_to_xlsx(var_array_tp, object_name, column_names, result_file + '_var.xlsx', 'variance_results')

# Credits: https://stackoverflow.com/a/69258865
def save_timing_to_xlsx(time_list,exp_name,result_file = "time_results.xlsx"):
    time_array = np.array(time_list)
    # print(time_array)
    df = pd.DataFrame(time_array[:,1],
                      index=[time_array[:,0]],
                      columns=[exp_name])
    # print(df)
    if not os.path.exists(result_file):
        df.to_excel(result_file, sheet_name='timing_sheet')
    else:
        workbook = openpyxl.load_workbook(result_file)  # load workbook if already exists
        sheet = workbook['timing_sheet']  # declare the active sheet 
        # append the dataframe results to the current excel file
        for row in dataframe_to_rows(df, header = False, index=[time_array[:,0]]):
            sheet.append(row)
        workbook.save(result_file)  # save workbook
        workbook.close()  # close workbook

def read_data(DIR,sht_name,nmb_rows,nmb_rows_skip,columns):
    dataframe = pd.read_excel(DIR,sheet_name=sht_name,usecols=columns,nrows=nmb_rows,skiprows=nmb_rows_skip)

    # print(dataframe.values)
    
    return dataframe.values

def save_to_latex(DIR,sht_name,nmb_rows,nmb_rows_skip,columns):
    dataframe = pd.read_excel(DIR,sheet_name=sht_name,usecols=columns,nrows=nmb_rows,skiprows=nmb_rows_skip)
    print(dataframe)
    dataframe.to_latex('test.tex',index=False,float_format="%.2f")
    # dataframe.to_latex('test.tex',index=False)
    return


# DIR = "C:/Users/menth/Documents/Universiteit/2223/Master Thesis/Results2.0_learning_registration.xlsx"
# save_to_latex(DIR, "Averaged", 2, 90, "T:AA,AD")