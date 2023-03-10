import os
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows


# Credits: https://stackoverflow.com/a/69258865
def save_to_xlsx(array,object_name,result_file = "saved_results.xlsx"):
    array_tp = np.transpose(array)
    df = pd.DataFrame(array_tp,
                      index=[object_name],
                      columns=['RE (°)', 'RE (m)','RMSE (°)','RMSE (m)','MAE (°)','MAE (m)',
                               'Recall','R2','CD'])
    # print(df)
    if not os.path.exists(result_file):
        df.to_excel(result_file, sheet_name='results_sheet')
    else:
        workbook = openpyxl.load_workbook(result_file)  # load workbook if already exists
        sheet = workbook['results_sheet']  # declare the active sheet 
        # append the dataframe results to the current excel file
        for row in dataframe_to_rows(df, header = False, index=[object_name]):
            sheet.append(row)
        workbook.save(result_file)  # save workbook
        workbook.close()  # close workbook
