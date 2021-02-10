import numpy as np
import pandas as pd
import os

day = 'Day30'
Path1 = "//li.lumentuminc.net/data/MIL/SJMFGING/Optics/Planar/MWRS For HPL(Test Data)/" + \
        "MWR-2020/Q2020287 Hopilte NAVA 14XX aging/"
Path2 = Path1 + day + '/'
files_ls = os.listdir(Path2)
files_ls = [_ for _ in files_ls if _.endswith('.txt')]
files_ls = [_ for _ in files_ls if _[0].isdigit()]
files_paths = [Path2 + filename for filename in files_ls]

df1 = pd.DataFrame(columns=['FBG_ID', 'Test_Time', 'Reflectivity(%)'])
for i, file in enumerate(files_paths):
    file1 = open(file, 'r')
    Line = file1.readlines()
    file1.close()
    FBG_ID = int(Line[0].replace('FBG_ID=', '').replace('\n', ''))
    Reflectivity = float(Line[5].replace('Reflectivity(%)=', '').replace('\n', ''))
    Test_Time = Line[1].replace('Test_Time=', '').replace('\n', '')

    to_append = [FBG_ID, Test_Time, Reflectivity]
    a_series = pd.Series(to_append, index=df1.columns)
    df1 = df1.append(a_series, ignore_index=True)

df1 = df1.sort_values(by=['FBG_ID'])
df2 = pd.DataFrame(columns=['FBG_ID', 'Test_Time', 'Reflectivity(%)'])

for i in list(set(df1['FBG_ID'])):
    df_temp1 = df1[df1['FBG_ID'] == i]
    df_temp2 = df_temp1[df_temp1['Reflectivity(%)'] == min(df_temp1['Reflectivity(%)'])]
    df2 = df2.append(df_temp2)

df2.to_csv(Path1 + 'Summary/' + day + ' Summary.csv', index=False)
