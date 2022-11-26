import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# directory = '.\\demand\\'
directory = '.\\sources\\'
extension = 'csv'
# merged_csv = "Demand_Merged.csv"
merged_csv = "Sources_Merged.csv"

#find all csv files in the folder
#use glob pattern matching -> extension = 'csv'
#save result in list -> all_filenames

# all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

all_filenames = [i for i in glob.glob((directory + '*.{}').format(extension))]
print(all_filenames)

#Big DataFrame
all_df = pd.DataFrame()
tmp_sum = []

for file in all_filenames:
    #check if a csv file is empty
    if os. stat(file).st_size == 0:
        print(file)
        os.remove(file)
    #merge csv to one DataFrame
    else:
        if file != f'{directory}20200229.csv':
            df = pd.read_csv(file, engine="python", skipfooter=1)
            tmp = list(df.iloc[:,1:].transpose().sum())
            tmp_sum += tmp
            all_df = pd.concat([all_df, df], ignore_index=True)

all_df['Sum'] = tmp_sum

#Merge the DataFrame into a csv file
all_df.to_csv(merged_csv, index=False, encoding='utf-8-sig')

# IF DEMANDS
# time_series = all_df.iloc[:, 3]
# ENDIF

# IF SOURCES
time_series = all_df.loc[:, 'Sum']
# ENDIF

print(time_series.describe())

count = time_series.count()
step = count // 3

plt.plot(time_series)
plt.axvline(x=step, color='k', linestyle='--')
plt.axvline(x=2*step, color='k', linestyle='--')
plt.show()

# split dataframe into 3 years

tmp = []
tmp.append(list(time_series.iloc[0:step]))
tmp.append(list(time_series.iloc[step:(2*step)]))
tmp.append(list(time_series.iloc[(2*step):count]))

tmp = pd.DataFrame(tmp).transpose()

# Create correlation matrix
corrMat = tmp.corr()
print(corrMat)

print("Debugging Message")
