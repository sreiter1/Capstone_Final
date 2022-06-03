# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:21:21 2022

@author: sreit
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import statsmodels.api as sm
import warnings
import sqlite3
import commonUtilities
import analysis2
import IndicatorsAndFilters
import StockData

import datetime as dt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import time

import math
import tensorflow as tf
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Input
from keras import metrics
from keras import losses
from keras import optimizers
from keras import callbacks
from sklearn.metrics import mean_squared_error, mean_absolute_error


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from pylab import rcParams



warnings.filterwarnings('ignore')

    
    

class MLmodels:
    def __init__(self, dataBaseSaveFile = "./stockData.db", splitDate = "2020-01-01"):
        
        tf.config.list_physical_devices('GPU')
        self.DB = sqlite3.connect(dataBaseSaveFile)
        self._cur = self.DB.cursor()
        self._tickerList = []   # Empty list that gets filled with a list of tickers to be considered
        self._data = pd.DataFrame()
        self.tradingDateSet = []  # List of dates in YYYY-MM-DD format that are trading dates in the database
        self.dailyTableNames = ["alpha", "yahoo"]
        self.splitDate = pd.to_datetime(splitDate)
        
        self.validate = commonUtilities.validationFunctions()
        
        # Converts a user string to the names of tables, and for 'ticker_symbol_list'
        self._tickerConversionTable = commonUtilities.conversionTables.tickerConversionTable
        
        # Converts user unput to the columns in the table.  Provides a filter to 
        # prevent database corruption.
        self._dailyConversionTable = commonUtilities.conversionTables.dailyConversionTable
        
        self.indicatorList = commonUtilities.conversionTables.indicatorList
        
        self.analysis = analysis2.analysis()
        self.indicators = IndicatorsAndFilters.filterData(dataBaseSaveFile = "stockData.db")
        self.stockdata = StockData.getAlphaVantageData()
        
        self.testX_out  = [] # for the LSTM model
        self.testYr_out = [] # for the LSTM model
        self.testYc_out = [] # for the LSTM model
        
    
    
    
    def LSTM_load(self, modelToLoad = "", learnRate = -1):
        CWDreset = os.getcwd()
        
        if modelToLoad == "":
            os.chdir("LSTM")
            os.chdir(max(os.listdir()))
            folderName = os.getcwd() + "\\"
            fileList = os.listdir()
            modelToLoad = fileList[0]
            for file in fileList:
                if "lstm_model" in file and file > modelToLoad:
                    modelToLoad = file
                if ".csv" in file:
                    prevTrainingData = file
                    
            self.modelToLoad = folderName + modelToLoad
            self.prevTrainingData = folderName + prevTrainingData
            
        else:
            folders = modelToLoad.split("\\")[:-1]
            self.prevTrainingData = "\\".join(folders) + "\\"
            os.chdir(self.prevTrainingData)
            fileList = os.listdir()
            for file in fileList:
                if ".csv" in file:
                    self.prevTrainingData += file
                    break
        
        os.chdir(CWDreset)
        
        print("Loading file: " + self.modelToLoad)
        try:
            self.lstm_model = load_model(self.modelToLoad, compile = False)
            
            
        except:
            raise ValueError("Failed to load lstm model.  Check ./LSTM/ to verify that models exist.")
        
        
        print("Loading file: " + self.prevTrainingData)
        try: 
            trainHist = pd.read_csv(self.prevTrainingData, header = 13)
        except:
            print("Failed to load previous training data.  Check ./LSTM/ to verify that training_data.csv is present.")
        
        if learnRate == -1:
            learnRate = trainHist["learn_rate"][-1]
        else:
            learnRate = 0.05
            
        self.compileLSTM(learnRate = learnRate)
        return trainHist
    
    
    
    
    def LSTM_eval(self):
        assert hasattr(self, "lstm_model"), "LSTM Model missing.  Train a new model with LSTM_train() or load a model with LSTM_load()."        
        
        self.testX_out, self.testYr_out, self.testYc_out = [], [], []
        
        for ticker in self.analysis._tickerList:
            trainX, trainYr, trainYc, testX, testYr, testYc = self.getLSTMTestTrainData(ticker = ticker)
            self.testX_out.append(testX)
            self.testYr_out.append(testYr)
            self.testYc_out.append(testYc)
            
            
        
        pred = []
        for i in range(len(self.testX_out)):
            pred.append(self.lstm_model.predict(self.testX_out[i]))
        
        evaluation = []
        for i in range(len(self.testX_out)):
            evaluation.append(self.lstm_model.evaluate(self.testX[i], [self.testYr_out[i], self.testYc_out[i]]))
        
        return pred, evaluation
    
    
    
    
    def LSTM_train(self, look_back = 120, 
                   EpochsPerTicker = 1,
                   fullItterations = 20,
                   tickerList = [], 
                   randomSeed = int(time.time()*100 % (2**32-1)), 
                   trainSize = -1, 
                   trainDate = "01-01-2020",
                   loadPrevious = True):
        
        np.random.seed(randomSeed)
        
        startTime = dt.datetime.now()
        self.trainingTimes   = []
        self.trainingHistory = []
        self.testingHistory  = []
        trainHistKeys = ['learn_rate', 'val_loss', 'val_out_reg_loss', 'val_out_cat_loss', 
                         'val_out_reg_mse', 'val_out_reg_mape', 'val_out_cat_auc', 
                         'val_out_cat_catAcc', 'val_out_cat_TP', 'val_out_cat_TN', 
                         'val_out_cat_FP', 'val_out_cat_FN', 'loss', 'out_reg_loss', 
                         'out_cat_loss', 'out_reg_mse', 'out_reg_mape', 'out_cat_auc', 
                         'out_cat_catAcc', 'out_cat_TP', 'out_cat_TN', 'out_cat_FP', 
                         'out_cat_FN']
        
        
        # Save file for the training data
        if loadPrevious:
            self.LSTM_load()
            prevItter = int(self.modelToLoad.split("_")[-1].split(".")[0])
            dataFile  = open(self.prevTrainingData, 'a')
            folderName = "\\".join(self.prevTrainingData.split("\\")[:-1]) + "\\"
        else:
            self.createLSTMNetwork(look_back = look_back)
            prevItter = 0
            folderName = "./LSTM/" + str(startTime.replace(microsecond=0)) + "/"
            folderName = folderName.replace(":", ".")
            os.makedirs(folderName)
            saveString = folderName + "training_data.csv"
            dataFile = open(saveString, 'w')
            dataFile.write("Saved Metrics:\n"
                           "MeanSquaredError = 'mse'\n" + 
                           "MeanAbsolutePercentageError = 'mape'\n" + 
                           "AUC = 'auc'\n" + 
                           "TruePositives = 'TP'\n" + 
                           "TrueNegatives = 'TN'\n" + 
                           "FalsePositives = 'FP'\n" + 
                           "FalseNegatives = 'FN'\n" + 
                           "CategoricalAccuracy = 'catAcc'\n" + 
                           "-------------------------------------\n\n\n")
            
            # looks weird, but works when loaded.  Training performance appears on the line above 
            # all the column headers, and so the 'ticker', 'itteration', and 'run time'
            # will line up and get loaded into pandas when the data is loaded
            dataFile.write(",,,training performance\nticker,itteration,run time,") 
            for key in trainHistKeys:
                dataFile.write(key + ",")
            dataFile.write("\n")
            dataFile.flush()
            
        
        
        if tickerList == []:
            tickerList = self.analysis._tickerList
            
        if tickerList == []:
            self.analysis.filterStocksFromDataBase(dailyLength = 1250, 
                                                    maxDailyChange = 50, 
                                                    minDailyChange = -50, 
                                                    minDailyVolume = 500000)
            tickerList = ["ZNGA"] # self.analysis._tickerList
        
        print(str(len(tickerList)).rjust(6) + "  Tickers selected by the filter.             ")
        
        
        # save the list of tickers for viewing later
        stringlist = []
        self.lstm_model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        
        if not loadPrevious:
            saveString = folderName + "tickerList - " + str(len(tickerList)) + ".txt"
            saveString = saveString.replace(":", ".")
            tickerFile = open(saveString, 'w')
            tickerFile.write(short_model_summary)
            tickerFile.write("\n\n-------------------------------------\n\n\nTickers included for training:\n\n")
            tickerFile.write(str(tickerList))
            tickerFile.close()
        
        
        for itteration in range(fullItterations):
            
            tickerCounter = 0
            tickerTotal   = str(len(tickerList))
            
            for ticker in tickerList:
                tickerCounter += 1
                
                trainX, trainYr, trainYc, testX, testYr, testYc = self.getLSTMTestTrainData(look_back = look_back,
                                                                                            ticker = ticker, 
                                                                                            trainSize = trainSize, 
                                                                                            trainDate = trainDate)
                
                print("\nTraining on " + ticker.rjust(6) + "  (" + str(tickerCounter) + " of " + tickerTotal \
                      + "),  Itteration (" + str(itteration + 1) + " of " + str(fullItterations) \
                      + "),  Elapsed Time: " + str(dt.datetime.now().replace(microsecond=0) - startTime.replace(microsecond=0)) \
                      + ",   Train/Test size: " + str(len(trainX)) + "/" + str(len(testX)) + "                  ")
                
                
                #---------------------------------------------------
                # Train the model
                
                self.lstm_model.reset_states()
                trainHist = self.lstm_model.fit(trainX, [trainYr, trainYc], 
                                                epochs = EpochsPerTicker, 
                                                verbose = 1, 
                                                validation_split = 0.2)
                
                #--------------------------------------------------
                # store the training information in memory
                
                self.trainingHistory.append( [ticker, itteration, trainHist.history]  )
                self.trainingTimes.append(   [ticker, itteration, dt.datetime.now()]  )
                
                #--------------------------------------------------
                # store the training information to disk
                     
                for i in range(len(trainHist.history["loss"])):
                    if i == 0:
                        dataString = ticker + "," + \
                                     str(itteration + prevItter)+ "," + \
                                     str(dt.datetime.now()).replace(" ","_") + ","
                    else:
                        dataString += ",,,"
                        
                    for key in trainHistKeys:
                        dataString += str(trainHist.history[key][i]) + ","
                        
                    dataString = dataString[:-1] + "\n"
                        
                
                dataFile.write(dataString)
                dataFile.flush()
                
            
            #--------------------------------------------------
            # Save and evalueate the model
            
            saveString = folderName + "lstm_model_" + str(itteration + 1 + prevItter).zfill(3) + ".h5"
            self.lstm_model.save(saveString)
            
            # evaluation = self.lstm_model.evaluate(testX_out, [testYr_out, testYc_out])
            # print("Final Results of Training:  " + str(evaluation))
        
        dataFile.close()
        return 
    
    
    
    def createLSTMNetwork(self, look_back):
        inLayer = Input(shape = (look_back, 7))
        hidden1 = LSTM(60,   name='LSTM1',   activation = "sigmoid", return_sequences=True, )(inLayer)
        hidden2 = LSTM(30,   name='LSTM2',   activation = "sigmoid")(hidden1)
        hidden3 = Dense(30,  name='dense2',  activation = "relu"   )(hidden2)
        outReg  = Dense(2,   name='out_reg', activation = "linear" )(hidden3)
        outCat  = Dense(1,   name='out_cat', activation = "sigmoid")(hidden3)
        
        self.lstm_model = Model(inputs=inLayer, outputs=[outReg, outCat])
        self.compileLSTM()
        
        print("---LSTM model built---\n")
        self.lstm_model.summary()
        
        
        
        
    def compileLSTM(self, learnRate = 0.05):
        opt = optimizers.Adam(learning_rate = learnRate)
        self.lstm_model.compile(optimizer = opt,
                                loss = {"out_reg" : losses.MeanAbsoluteError(),
                                        "out_cat" : losses.BinaryCrossentropy(label_smoothing=0.2)},
                                metrics = {"out_reg": [metrics.MeanSquaredError(name = "mse"), 
                                                       metrics.MeanAbsoluteError(name = "mae"),
                                                       metrics.MeanAbsolutePercentageError(name = "mape")],
                                           "out_cat": [metrics.AUC(name = "auc"),
                                                       metrics.CategoricalAccuracy(name = "catAcc"), 
                                                       metrics.TruePositives(name = "TP"),
                                                       metrics.TrueNegatives(name = "TN"),
                                                       metrics.FalsePositives(name = "FP"),
                                                       metrics.FalseNegatives(name = "FN")]})
        
        self.lstm_model.add_metric(mod.lstm_model.optimizer.learning_rate, name = "learn_rate")
    
    
    
    
    def multiFeatureScalingArray(self, big, small, numFeatures):
        out = pd.DataFrame()
        
        for i in range(numFeatures):
            out[str(i)] = [big, small]
        
        return out
    
    
    
    
    def splitLSTMData(self, inputs, outputs_r, outputs_c, look_back = 120, trainSize = 0.8):
        # takes 2D np array of data with axes of time (i.e. trading days) and features,
        # and returns a 3D np array of batch, time, features
        
        lenInput = len(inputs)
        
        
        if trainSize < 1 and trainSize > 0:
            lenTrain = int(lenInput * trainSize)
            if trainSize < 0.5:
                warnings.warn("testSize is declared to be less than half of the dataset; results may not be useful.")
        elif trainSize > 1:
            lenTrain = trainSize
        else:
            raise ValueError("test size < 0; this is not valid.")
            
            
        trainX  = inputs[:lenTrain]
        trainYr = outputs_r[:lenTrain]
        trainYc = outputs_c[:lenTrain]
                           
        testX   = inputs[(lenTrain+1):]
        testYr  = outputs_r[(lenTrain+1):]
        testYc  = outputs_c[(lenTrain+1):]
        
        return trainX, trainYr, trainYc, testX, testYr, testYc
    
    
    
    
    def normLSTMData(self, shaped_data_in, returnNormalizer = False):
        # normalizes data based on specific inputs
        # data_in should be a 3D array of shape batch, time steps, features
        
        data_out = []
        try:
            features = len(shaped_data_in[0][0])
        except:
            features = 1
            
        normalizer = MinMaxScaler()
        
        for i in range(len(shaped_data_in)):
            
            print("\rNormalising itteration  " + str(i) + "                      ", end = "\r")
            fitData = self.multiFeatureScalingArray(np.amax(shaped_data_in[i]), 
                                                    np.amin(shaped_data_in[i]), 
                                                    features)
            
            normalizer.fit(fitData)
            a = normalizer.transform(shaped_data_in[i])
            data_out.append(a)
        
        if returnNormalizer:
            return np.array(data_out), normalizer
        else:
            return np.array(data_out)
    
    
    
    
    def reshapeLSTMData(self, data_in, io, look_back = 120):
        # takes 2D np array of data with axes of time (i.e. trading days) and features,
        # and returns a 3D np array of batch, time, feature
        
        shaped_data = []
        lenInput = len(data_in)
        
        if io == "input":
            for i in range(lenInput - look_back - 1):
                print("\rShaping itteration  " + str(i) + "                      ", end = "\r")
                a = data_in[i:(i+look_back)]
                shaped_data.append(a)
                data_out = self.normLSTMData(np.array(shaped_data))
                
        else:
            for i in range(lenInput - look_back - 1):
                print("\rShaping itteration  " + str(i) + "                      ", end = "\r")
                a = data_in[i+look_back]
                shaped_data.append(a)
                data_out = self.normLSTMData(np.array(shaped_data))
        
        return data_out
        
        
    
    
    
    def getLSTMTestTrainData(self, 
                             look_back = 120,
                             ticker = "", 
                             trainSize = -1, 
                             trainDate = "01-01-2020"):
        
        loadedData, t = self.analysis.loadFromDB(tickerList = [ticker],
                                                 indicators = ["MA20", "OBV", "IDEAL"],
                                                 extras = ["HIGH", "LOW", "ADJRATIO", "VOLUME",
                                                           "IDEAL_HIGH", "IDEAL_LOW", "IDEAL_TRIG"])
        
        loadedData['recordDate'] = pd.to_datetime(loadedData['recordDate'])
        loadedData.sort_values(by = ["ticker_symbol", "recordDate"], ascending=True, inplace=True)
        
        # set the training size to be either a decimal from the funciton input, or to a all records before a set date
        if trainSize == -1:
            trainLen = len(loadedData.loc[loadedData["recordDate"] < pd.to_datetime(trainDate)])
        elif 0 < trainSize and trainSize < 1:
            trainLen = int(trainSize * len(loadedData["recordDate"]))
        elif trainSize > 1:
            trainLen = min(int(trainSize), len(loadedData["recordDate"]-1))
        else:
            trainLen = int(0.8 * len(loadedData["recordDate"]))
            
        
        # ensure that the adjustment ratio does not cause a div-by-0 error
        assert min(loadedData["adjustment_ratio"]) > 0, "\n\n  ERROR: adjustment ratio has 0-value.  Verify correct input data.  Ticker = " + ticker
        
        
        # create the input frames that will be translated to numpy arrays
        inputFrame  = pd.DataFrame()
        outputFrame = pd.DataFrame()
        
        inputFrame["open" ] = [o/a for o,a in zip(loadedData["open"],  loadedData["adjustment_ratio"])]
        inputFrame["high" ] = [h/a for h,a in zip(loadedData["high"],  loadedData["adjustment_ratio"])]
        inputFrame["low"  ] = [l/a for l,a in zip(loadedData["low"],   loadedData["adjustment_ratio"])]
        inputFrame["close"] = [c/a for c,a in zip(loadedData["close"], loadedData["adjustment_ratio"])]
        inputFrame["ma20" ] = loadedData["mvng_avg_20"]
        inputFrame["obv"  ] = loadedData["on_bal_vol"]
        inputFrame["vol"  ] = loadedData["volume"] 
        
        outputFrame["high"] = loadedData["ideal_high"]
        outputFrame["low" ] = loadedData["ideal_low" ]
        outputFrame["trig"] = [1 if t==1 else 0 for t in loadedData["ideal_return_trig"]]
        
        
        # ----------------------------------------------
        # scale the inputs and outputs.  Pricing data is separated from the OBV data
        # Also need to convert to numpy array with dimmensions:
        # [batch (i.e. time series), timesteps (i.e. trade dates), features (i.e. prices)]
        # ----------------------------------------------
        
        # Reshape and normalize the data.  Inputs/outputs are broken into segments
        # that are normalized together to preserve certain relationships between them
        # and then recombined as inputs and the 2 separate output predictions.  
        # Once the data has been shaped and scaled, it is split into test and train
        # sets.
        input_prc   = self.reshapeLSTMData(inputFrame[["open", "high", "low", "close", "ma20"]].to_numpy(), "input", look_back = look_back)
        input_obv   = self.reshapeLSTMData(inputFrame[["obv"]].to_numpy().reshape(-1,1), "input", look_back = look_back)
        input_vol   = self.reshapeLSTMData(inputFrame[["vol"]].to_numpy().reshape(-1,1), "input", look_back = look_back)
        outputs_reg = self.reshapeLSTMData(outputFrame[["high", "low"]].to_numpy(),      "output", look_back = look_back)
        outputs_cat = self.reshapeLSTMData(outputFrame["trig"].to_numpy().reshape(-1,1), "output", look_back = look_back)
        
        input_data = np.concatenate((input_prc,  input_obv), axis = 2)
        input_data = np.concatenate((input_data, input_vol), axis = 2)
        
        trainX, trainYr, trainYc, testX, testYr, testYc = self.splitLSTMData(input_data, outputs_reg, outputs_cat, look_back = look_back, trainSize = trainLen)
        
        
        return trainX, trainYr, trainYc, testX, testYr, testYc
    
    
    
    
    def Trees(self, ticker = "", future_days = 100):
        # https://www.kaggle.com/code/rishidamarla/stock-market-prediction-using-decision-tree/notebook
        priceData, t = self.analysis.loadFromDB(tickerList = [ticker],
                                                indicators = ["MA20", "OBV", "IDEAL"],
                                                extras = ["HIGH", "LOW", "ADJRATIO", "VOLUME",
                                                          "IDEAL_HIGH", "IDEAL_LOW", "IDEAL_TRIG"])
        
        priceData = priceData[priceData["ticker_symbol"] == ticker]
        priceData = priceData.drop(["ticker_symbol"], 1)
        priceData = priceData.drop(["recordDate"], 1)
        
        priceData["Prediction"] = priceData["adj_close"].shift(-future_days)
        
        X = np.array(priceData.drop(["Prediction"], 1))[:-future_days]
        y = np.array(priceData["Prediction"])[:-future_days]
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        tree = DecisionTreeRegressor().fit(x_train, y_train)
        
        x_future = priceData.drop(["Prediction"], 1)[:-future_days]
        x_future = x_future.tail(future_days)
        x_future = np.array(x_future)
        
        tree_prediction = tree.predict(x_future)
        
        predictions = tree_prediction
        valid = priceData[X.shape[0]:]
        valid["Predictions"] = predictions
        
        
        plt.figure.Figure(figsize=(16,8))
        plt.pyplot.title("Model")
        plt.pyplot.xlabel("Days")
        plt.pyplot.ylabel("Close Price USD ($)")
        plt.pyplot.plot(priceData["adj_close"])
        plt.pyplot.plot(valid[["adj_close", "Predictions"]])
        plt.pyplot.legend(["Original", "Valid", "Predicted"])
        plt.pyplot.show()
        
        return tree
    
    
    
    
    def linearRegression(self, ticker = "", future_days = 100):
        # https://www.kaggle.com/code/rishidamarla/stock-market-prediction-using-decision-tree/notebook
        priceData, t = self.analysis.loadFromDB(tickerList = [ticker],
                                                indicators = ["MA20", "OBV", "IDEAL"],
                                                extras = ["HIGH", "LOW", "ADJRATIO", "VOLUME",
                                                          "IDEAL_HIGH", "IDEAL_LOW", "IDEAL_TRIG"])
        
        
        priceData = priceData[priceData["ticker_symbol"] == ticker]
        priceData = priceData.drop(["ticker_symbol"], 1)
        priceData = priceData.drop(["recordDate"], 1)
        
        priceData["Prediction"] = priceData["adj_close"].shift(-future_days)
        
        X = np.array(priceData.drop(["Prediction"], 1))[:-future_days]
        y = np.array(priceData["Prediction"])[:-future_days]
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        lr = LinearRegression().fit(x_train, y_train)
        
        x_future = priceData.drop(["Prediction"], 1)[:-future_days]
        x_future = x_future.tail(future_days)
        x_future = np.array(x_future)
        
        lr_prediction = lr.predict(x_future)
        
        predictions = lr_prediction
        valid = priceData[X.shape[0]:]
        valid["Predictions"] = predictions
        
        plt.figure.Figure(figsize=(16,8))
        plt.pyplot.title("Model")
        plt.pyplot.xlabel("Days")
        plt.pyplot.ylabel("Close Price USD ($)")
        plt.pyplot.plot(priceData["adj_close"])
        plt.pyplot.plot(valid[["adj_close", "Predictions"]])
        plt.pyplot.legend(["Original", "Valid", "Predicted"])
        plt.pyplot.show()
        
        return lr
    
    
    
    
    def ARIMA(self, ticker = ""):
        # https://www.analyticsvidhya.com/blog/2021/07/stock-market-forecasting-using-time-series-analysis-with-arima-model/
        priceData, t = self.analysis.loadFromDB(tickerList = [ticker],
                                                indicators = ["MA20", "OBV", "IDEAL"],
                                                extras = ["HIGH", "LOW", "ADJRATIO", "VOLUME",
                                                          "IDEAL_HIGH", "IDEAL_LOW", "IDEAL_TRIG"])
                
        priceData = priceData[priceData["ticker_symbol"] == ticker]
        
        rcParams['figure.figsize'] = 10, 6
        df_log = np.log(priceData["adj_close"])
        moving_avg = df_log.rolling(12).mean()
        std_dev = df_log.rolling(12).std()
        
        train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
        
        model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                          test='adf',       # use adftest to find optimal 'd'
                          max_p=3, max_q=3, # maximum p and q
                          m=1,              # frequency of series
                          d=None,           # let model determine 'd'
                          seasonal=False,   # No Seasonality
                          start_P=0, 
                          D=0, 
                          trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=True)
        
        print(model_autoARIMA.summary())
        model_autoARIMA.plot_diagnostics(figsize=(15,8))
        plt.show()
        
        model = ARIMA(train_data, order=(1,1,2))  
        fitted = model.fit(disp=-1)  
        print(fitted.summary())
        
        fc, se, conf = fitted.forecast(321, alpha=0.05)
        
        # Make as pandas series
        fc_series = pd.Series(fc, index=test_data.index)
        lower_series = pd.Series(conf[:, 0], index=test_data.index)
        upper_series = pd.Series(conf[:, 1], index=test_data.index)
        # Plot
        plt.figure(figsize=(10,5), dpi=100)
        plt.plot(train_data, label='training data')
        plt.plot(test_data, color = 'blue', label='Actual Stock Price')
        plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
        plt.fill_between(lower_series.index, lower_series, upper_series, 
                         color='k', alpha=.10)
        plt.title('ARCH CAPITAL GROUP Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('ARCH CAPITAL GROUP Stock Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()
        
        mse = mean_squared_error(test_data, fc)
        print('MSE: '+str(mse))
        mae = mean_absolute_error(test_data, fc)
        print('MAE: '+str(mae))
        rmse = math.sqrt(mean_squared_error(test_data, fc))
        print('RMSE: '+str(rmse))
        mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
        print('MAPE: '+str(mape))
        
        return model
        
        
        
        



    
if __name__ == "__main__":
    
    if 'mod' not in locals():
        mod = MLmodels()
        print("\n---- New instance of MLmodels created. ----")
    
    mod.analysis.filterStocksFromDataBase(dailyLength = 1250, 
                                          maxDailyChange = 50, 
                                          minDailyChange = -50, 
                                          minDailyVolume = 500000)
    
    
    # tree = mod.Trees("A")
    # lr_model = mod.linearRegression("A")
    # arima_model = mod.ARIMA("A")
    
    
    mod.LSTM_train(loadPrevious = False, EpochsPerTicker = 20, fullItterations = 1)
    # data = mod.LSTM_load()
    # lstm_pred = mod.LSTM_test()

    
    
    
    
    
    
    
    
    
    