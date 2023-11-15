import pandas as pd
import os
import openpyxl


#import and basic cleaning
file_path=os.path.join(os.getcwd(), 'MVE_Assignment_DataSet.xlsx')
data = pd.read_excel(file_path,header=1)
data.rename(columns={'Unnamed: 0': 'year',
                     'EN.ATM.METH.AG.KT.CE' : 'methane' ,
                     'EN.ATM.NOXE.AG.KT.CE' :  'nox' ,
                     'AG.PRD.CROP.XD' :  'crop_production' ,
                     'EN.ATM.CO2E.KT' :  'co2',
                     'mean+tmp' :  'mean_temp',
                     'mean_pre': 'mean_rainfall',
                     'SP.POP.TOTL' : 'population',
                     'NV.AGR.TOTL.KD' :  'agricultural_GDP'     
                     }, inplace=True)