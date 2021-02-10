import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import os

path = "//li.lumentuminc.net/data/MIL\SJMFGING/Optics/Planar/MWRS For HPL(Test Data)/" + \
       "MWR-2020/Q2020287 Hopilte NAVA 14XX aging/Summary/"
files_ls = os.listdir(path)
files_ls = [_ for _ in files_ls if _.startswith('Time')] + [_ for _ in files_ls if _.startswith('Day')]
files_paths = [path + filename for filename in files_ls]

n = 2  # polyfit order
company = 'LITE'
[SN1, SN2, SN3] = [[3, 4, 5, 6, 7, 8], [11, 12, 14, 16], [19, 20, 21, 22, 23, 24]] if \
    company == 'LITE' else [[1, 2], [9], [17, 18]]
Serial_Number = SN1 + SN2 + SN3  # SN
Sample_sizes = [len(SN1), len(SN2), len(SN3)]
Serial_T = [125] * Sample_sizes[0] + [180] * Sample_sizes[1] + [250] * Sample_sizes[2]  # Temperature in C
Serial_T = [_ + 297 for _ in Serial_T]  # Temperature in K

# read csv files
data = []
for i, file_path in enumerate(files_paths):
    df = pd.read_csv(file_path, index_col=0)
    df = df.drop([x for x in df.index if x not in Serial_Number])  # drop samples if it is not shown in Serial_Number
    df['Test_Time'] = df['Test_Time'].map(lambda x: x[:-5].replace('_', ''))  # remove '_'
    df['Test_Time'] = pd.to_datetime(df['Test_Time'], format='%Y%m%d%H%M', errors='ignore')  # change to date time
    df = df[['Test_Time', 'Reflectivity(%)']]
    data.append(df.to_numpy())
data = np.array(data)
data[:, :, 0] = data[:, :, 0] - np.min(data[:, :, 0], axis=0)  # Convert time to time difference
sorted_index = np.argsort(data[:, 0, 0])
data = data[sorted_index, :, :]  # Sort by time difference

#  convert timedelta to hours
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data[i, j, 0] = data[i, j, 0].total_seconds() / 60 / 60  # from seconds to hours

# Calculate NICC
NICC = np.arctanh(np.sqrt(data[:, :, 1].astype(float) / 100))
NICC = NICC / np.max(NICC[:, :], axis=0)

# Time delta
# time_delta = numpy.matlib.repmat(np.min(data[:, :, 0], axis=1), 18, 1).transpose()
time_delta = data[:, :, 0]

#  plot reflection rate & NICC
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
Colors = ['r'] * Sample_sizes[0] + ['g'] * Sample_sizes[1] + ['b'] * Sample_sizes[2]
E0 = []  # initialize E = KT*ln(t) in eV
for i in range(data.shape[1]):
    ax1.plot(time_delta[:, i], data[:, i, 1], 'o-', color=Colors[i])
    ax2.plot(time_delta[:, i], NICC[:, i], 'o-', color=Colors[i])

    # Calculate E = KT*ln(t) in eV
    e0 = 8.617e-5 * Serial_T[i] * np.log(time_delta[1:, i].astype(float) * 60 * 60)
    E0.append(e0)
    ax3.plot(e0, NICC[1:, i], 'o-', color=Colors[i])

ax1.set_title('R vs. time, 250C, 180C and 125C heating')
ax1.set_xlabel('Heating time (hours)')
ax1.set_ylabel('Reflectivity (%)')
ax1.grid()
fig1.savefig(path + company + ' - Reflectivity vs time.png')

ax2.set_title('NICC vs. time, 250C, 180C and 125C heating')
ax2.set_xlabel('Heating time (hours)')
ax2.set_ylabel('NICC')
ax2.grid()
fig2.savefig(path + company + ' - NICC vs time.png')

ax3.set_title('')
ax3.set_xlabel('K$_{B}$T.ln(t) (eV)')
ax3.set_ylabel('NICC')
ax3.grid()
fig3.savefig(path + company + ' - NICC vs eV.png')

# Adjust ln(k) to get the best fitting line (with LMS method)
E0 = np.array(E0)
y = NICC[1:, :].transpose().reshape(-1)
Error = []
range_lnk = np.arange(10, 100, 0.1)
for i in range_lnk:
    Ed = E0 + 8.617e-5 * np.array(Serial_T)[:, None] * i
    x = Ed.reshape(-1)
    p = np.poly1d(np.polyfit(x, y, n))
    error = np.sum((np.polyval(np.polyfit(x, y, n), x) - y) ** 2)  # LMS
    Error.append(error)

lnk = range_lnk[Error.index(min(Error))]  # ln(k)
Ed = E0 + 8.617e-5 * np.array(Serial_T)[:, None] * lnk
x = Ed.reshape(-1)
p = np.poly1d(np.polyfit(x, y, n))
t = np.linspace(min(x), max(x), 200)
fig4, ax4 = plt.subplots()
ax4.plot(x, y, 'o', t, p(t), '-')
ax4.set_title(company + ' - Master curve')
ax4.set_xlabel('Ed (eV)')
ax4.set_ylabel('NICC')
ax4.grid()
fig4.savefig(path + company + ' - Master curve.png')

year = np.arange(1, 26, 1)  # [year]
T_op = 70 + 273
Ed_new = 8.617e-5 * T_op * (np.log(year * 365 * 24 * 60 * 60) + lnk)
NICC_new = p(Ed_new)
R0 = 0.03  # 3% reflection
X0 = R0 ** 0.5
R = (((1 + X0) / (1 - X0)) ** NICC_new - 1) ** 2 / (((1 + X0) / (1 - X0)) ** NICC_new + 1) ** 2
fig5, ax5 = plt.subplots()
ax5.plot(year, (R / R[0] - 1) * 100)
ax5.set_title('')
ax5.set_xlabel('Time (years)')
ax5.set_ylabel('Relative Reflectivity decay (%)')
ax5.set_title(company + ' - Reflectivity Prediction')
ax5.grid()
fig5.savefig(path + company + ' - Reflectivity Prediction.png')
