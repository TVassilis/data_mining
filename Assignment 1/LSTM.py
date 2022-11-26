import numpy
import pandas as pd
import glob
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import math

directory = '.\\sources\\'
extension = 'csv'

merged_csv = 'Alternative_Sources_Merged_2.csv'

def create_dataset(lstm_df, look_back=1):
    dataX, dataY = [], []
    for i in range(len(lstm_df)-look_back-1):
        a = lstm_df[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(lstm_df[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

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

            if 'Natural gas' in df:
                df.rename(columns = {'Natural gas': 'Natural Gas', 'Large hydro': 'Large Hydro'}, inplace=True)
            df = df.loc[:, df.columns.drop(['Coal', 'Nuclear', 'Biogas', 'Batteries', 'Imports', 'Other', 'Natural Gas'])]
            tmp = list(df.iloc[:,1:].transpose().sum())
            # print(df.iloc[:,1:])
            # exit()
            tmp_sum += tmp
            all_df = pd.concat([all_df, df], ignore_index=True)

all_df['Sum'] = tmp_sum

#Merge the DataFrame into a csv file
all_df.to_csv(merged_csv, index=False, encoding='utf-8-sig')

altsources_df = all_df['Sum']
print(altsources_df)
m1 = altsources_df.mean()
all_df = altsources_df.fillna(m1)


demands_df = pd.read_csv('Demand_Merged.csv')
demands_df = demands_df['Current demand']

m2 = demands_df.mean()
demands_df = demands_df.fillna(m2)

lstm_df = demands_df.subtract(all_df)
lstm_df.dropna(inplace=True)

# standardize the dataset
scaler = StandardScaler()
dataset = scaler.fit_transform(numpy.array(lstm_df).reshape(-1, 1))

# dataset = zscore(lstm_df).reshape(-1, 1)

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = numpy.array(dataset[0:train_size,:]), numpy.array(dataset[train_size:len(dataset),:])
print(len(train), len(test))

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(8, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=3, batch_size=50)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

print(trainY, trainPredict, sep='\n\n')
print(testY, testPredict, sep='\n\n')


