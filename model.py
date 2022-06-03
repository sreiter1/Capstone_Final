# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:21:21 2022

@author: sreit
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import warnings
import commonUtilities
import analysis2
import IndicatorsAndFilters
import StockData
import pickle

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
from keras.layers import LSTM
from keras.layers import Input
from keras import metrics
from keras import optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from pylab import rcParams



warnings.filterwarnings('ignore')

    
    

class MLmodels:
    def __init__(self, dataBaseSaveFile = "./stockData.db", 
                 dataBaseThreadCheck = True,
                 splitDate = "2020-01-01"):
        
        tf.config.list_physical_devices('GPU')
        self._tickerList = []   # Empty list that gets filled with a list of tickers to be considered
        self._data = pd.DataFrame()
        self.tradingDateSet = []  # List of dates in YYYY-MM-DD format that are trading dates in the database
        self.splitDate = pd.to_datetime(splitDate)
        
        self.validate = commonUtilities.validationFunctions()
        
        # Converts a user string to the names of tables, and for 'ticker_symbol_list'
        self._tickerConversionTable = commonUtilities.conversionTables.tickerConversionTable
        
        # Converts user unput to the columns in the table.  Provides a filter to 
        # prevent database corruption.
        self._dailyConversionTable = commonUtilities.conversionTables.dailyConversionTable
        
        self.indicatorList = commonUtilities.conversionTables.indicatorList
        
        self.analysis   = analysis2.analysis(dataBaseSaveFile = dataBaseSaveFile, 
                                             dataBaseThreadCheck = dataBaseThreadCheck)
        self.indicators = IndicatorsAndFilters.filterData(dataBaseSaveFile = dataBaseSaveFile, 
                                                          dataBaseThreadCheck = dataBaseThreadCheck)
        self.stockdata  = StockData.getAlphaVantageData(dataBaseSaveFile = dataBaseSaveFile, 
                                                        dataBaseThreadCheck = dataBaseThreadCheck)
        
        self.testX_out  = [] # for the LSTM model
        self.testYr_out = [] # for the LSTM model
        self.testYc_out = [] # for the LSTM model
        
    
    
    
    def LSTM_load(self, modelToLoad = ""):
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
            self.compileLSTM()
            
            
        except:
            raise ValueError("Failed to load lstm model.  Check ./LSTM/ to verify that models exist.")
        
        
        print("Loading file: " + self.prevTrainingData)
        try: 
            trainHist = pd.read_csv(self.prevTrainingData, header = 13)
        except:
            print("Failed to load previous training data.  Check ./LSTM/ to verify that training_data.csv is present.")
        
        return trainHist
    
    
    
    
    def LSTM_eval(self, ticker = "",
                  savePlt = False, 
                  evaluate = True, 
                  predLen = 100, 
                  plotWindow = 600,
                  timestampstr = ""):
        
        assert hasattr(self, "lstm_model"), "LSTM Model missing.  Train a new model with LSTM_train() or load a model with LSTM_load()."        
        
        if evaluate:
            trainSize = -1
        else:
            trainSize = 0.9
        
        trainX, trainYr, trainYc, testX, testYr, testYc = self.getLSTMTestTrainData(ticker = ticker, 
                                                                                    trainSize = trainSize)
        
        
        evaluation = self.lstm_model.evaluate(testX, [testYr, testYc])
        
        
        prediction = self.lstm_model.predict(testX)
        
        
        return prediction, evaluation, testX, [testYr, testYc]
    
    
    
    
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
                                     str(dt.datetime.now()) + ","
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
        # Model 1:
        # inLayer = Input(shape = (look_back, 7))
        # hidden1 = LSTM(120,  name='LSTM',    activation = "sigmoid")(inLayer)
        # hidden2 = Dense(128, name='dense1',  activation = "relu"   )(hidden1)
        # hidden3 = Dense(128, name='dense2',  activation = "relu"   )(hidden2)
        # outReg  = Dense(2,   name='out_reg', activation = "linear" )(hidden3)
        # outCat  = Dense(2,   name='out_cat', activation = "softmax")(hidden3)
        
        # self.lstm_model = Model(inputs=inLayer, outputs=[outReg, outCat])
        # self.compileLSTM()
        
        
        
        # Model 2
        inLayer = Input(shape = (look_back, 7))
        hidden1 = LSTM(120,  name='LSTM',    activation = "sigmoid")(inLayer)
        hidden2 = Dense(128, name='dense1',  activation = "relu"   )(hidden1)
        outReg  = Dense(2,   name='out_reg', activation = "linear" )(hidden2)
        outCat  = Dense(2,   name='out_cat', activation = "softmax")(hidden2)
        
        self.lstm_model = Model(inputs=inLayer, outputs=[outReg, outCat])
        self.compileLSTM()
        
        print("---LSTM model built---\n")
        self.lstm_model.summary()
        
        
        
        
    def compileLSTM(self):
        opt = optimizers.Adam(learning_rate = 0.05)
        self.lstm_model.compile(optimizer = opt,
                                loss = {"out_reg" : "mean_squared_error", 
                                        "out_cat" : "categorical_crossentropy"},
                                metrics = {"out_reg": [metrics.MeanSquaredError(name = "mse"), 
                                                       metrics.MeanAbsolutePercentageError(name = "mape")],
                                           "out_cat": [metrics.AUC(name = "auc"),
                                                       metrics.CategoricalAccuracy(name = "catAcc"), 
                                                       metrics.TruePositives(name = "TP"),
                                                       metrics.TrueNegatives(name = "TN"),
                                                       metrics.FalsePositives(name = "FP"),
                                                       metrics.FalseNegatives(name = "FN")]},
                                loss_weights = {"out_reg": 1.0, "out_cat": 5.0})
    
    
    
    
    def getFitArray(self, big, small, features):
        out = pd.DataFrame()
        
        for i in range(features):
            out[str(i)] = [big, small]
        
        return out
    
    
    
    
    def splitLSTMData(self, inputs, outputs_r, outputs_c, look_back = 200, trainSize = 0.8):
        # takes 2D np array of data with axes of time (i.e. trading days) and features,
        # and returns a 3D np array of batch, time, features
        
        dataX, dataY_r, dataY_c = [], [], []
        lenInput = len(inputs)
        
        
        if trainSize < 1 and trainSize > 0:
            lenTrain = int(lenInput * trainSize)
            if trainSize < 0.5:
                warnings.warn("testSize is declared to be less than half of the dataset; results may not be useful.")
        elif trainSize > 1:
            lenTrain = trainSize
        else:
            raise ValueError("test size < 0; this is not valid.")
            
            
        for i in range(lenInput - look_back - 1):
            a = inputs[i:(i+look_back)]
            dataX.append(a)
            dataY_r.append(outputs_r[i + look_back])
            dataY_c.append(outputs_c[i + look_back])
            
        
        trainX = np.array(dataX[:lenTrain])
        trainYr = np.array(dataY_r[:lenTrain])
        trainYc = np.array(dataY_c[:lenTrain])
        
        testX = np.array(dataX[(lenTrain+1):])
        testYr = np.array(dataY_r[(lenTrain+1):])
        testYc = np.array(dataY_c[(lenTrain+1):])
        
        return trainX, trainYr, trainYc, testX, testYr, testYc
    
    
    
    
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
        elif 0 < trainSize and trainSize <= 1:
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
        
        # create scalers for the data
        inPriceNorm  = MinMaxScaler()
        inOBVNorm    = MinMaxScaler()
        inVolNorm    = MinMaxScaler()
        outPriceNorm = MinMaxScaler()
        
        # get the fit data to match the highest high and lowest low so
        # that all the pricing data is scaled together (relationships between 
        # features should be maintained).  Then append the OBV data.
        fitList = self.getFitArray(max(inputFrame["high"]), min(inputFrame["low"]), 5)  #should small (min) be 0?  or min('low')?
        inPriceNorm.fit(fitList)
        inputs_norm  = inPriceNorm.transform(inputFrame[["open", "high", "low", "close", "ma20"]])
        inputs_norm  = np.concatenate((inputs_norm, inOBVNorm.fit_transform(inputFrame[["obv"]]).reshape(-1,1)), axis = 1)
        inputs_norm  = np.concatenate((inputs_norm, inVolNorm.fit_transform(inputFrame[["vol"]]).reshape(-1,1)), axis = 1)
        
        
        # get the fit data to match the highest high and lowest low so
        # that all the pricing data is scaled together (relationships between 
        # features should be maintained).  Then append the trigger data.
        fitList = self.getFitArray(max(inputFrame["high"]), min(inputFrame["low"]), 2)
        outPriceNorm.fit(fitList)
        outputs_reg = outPriceNorm.transform(outputFrame[["high", "low"]])
        outputs_cat = OneHotEncoder(sparse = False).fit_transform(outputFrame["trig"].to_numpy().reshape(-1,1))
        
        # modify the inputs and outputs to match the LSTM expectations
        # and separeate into test and train sets
        trainX, trainYr, trainYc, testX, testYr, testYc = self.splitLSTMData(inputs_norm, outputs_reg, outputs_cat, look_back = look_back, trainSize = trainLen)
        self.testX_out.append(testX)
        self.testYr_out.append(testYr)
        self.testYc_out.append(testYc)
        
        return trainX, trainYr, trainYc, testX, testYr, testYc
    
    
    
    
    def Trees(self, ticker = "", 
              savePlt = False, 
              evaluate = True, 
              predLen = 100, 
              plotWindow = 600,
              timestampstr = ""):
        # https://www.kaggle.com/code/rishidamarla/stock-market-prediction-using-decision-tree/notebook
        loadedData, t = self.analysis.loadFromDB(tickerList = [ticker],
                                                indicators = ["MA20", "OBV", "IDEAL"],
                                                extras = ["HIGH", "LOW", "ADJRATIO", "VOLUME",
                                                          "IDEAL_HIGH", "IDEAL_LOW", "IDEAL_TRIG"])
        
        priceData = loadedData[loadedData["ticker_symbol"] == ticker]
        priceData = priceData.drop(["ticker_symbol"], 1)
        priceData = priceData.drop(["recordDate"], 1)
        
        priceData["Prediction"] = priceData["adj_close"].shift(-predLen)
        
        X = np.array(priceData.drop(["Prediction"], 1))[:-predLen]
        y = np.array(priceData["Prediction"])[:-predLen]
        
        if evaluate:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
            
            x_future = priceData.drop(["Prediction"], 1)[:-predLen]
            x_future = x_future.tail(predLen)
            predIndex = max(x_future.index)
            x_future = np.array(x_future)
            
            y_future = priceData["Prediction"][:-predLen]
            y_future = y_future.tail(predLen)
            y_future = np.array(y_future)
            
        else:
            x_train = X
            y_train = y
            
            x_future = priceData.drop(["Prediction"], 1)
            x_future = x_future.tail(predLen*2)
            predIndex = max(x_future.index) - predLen
            x_future = np.array(x_future)
        
        
        tree = DecisionTreeRegressor().fit(x_train, y_train)
    
        predictions = tree.predict(x_future)
        predictions = pd.DataFrame(predictions)
        predictions = pd.concat([predictions, pd.Series(list(range(predIndex, predIndex + len(predictions))))], axis = 1)
        predictions.columns = ["treePrediction","ind"]
        predictions = predictions.set_index("ind")
        
        valid = priceData[X.shape[0]:]
        
        plt.figure.Figure(figsize=(16,8))
        plt.pyplot.title("Model")
        plt.pyplot.xlabel("Days")
        plt.pyplot.ylabel("Close Price USD ($)")
        plt.pyplot.plot(priceData["adj_close"])
        plt.pyplot.plot(valid["adj_close"])
        plt.pyplot.plot(predictions)
        plt.pyplot.xlim([len(priceData)-plotWindow, len(priceData)+predLen])
        plt.pyplot.legend(["Original", "True", "Predicted"])
        
        if savePlt:
            plt.pyplot.savefig("./static/Tree_1_" + timestampstr + ".png")
        else:
            plt.pyplot.show()
        
        if evaluate:
            print("Linear Regression performance: ")
            mse = mean_squared_error(y_future, predictions["treePrediction"])
            print('MSE: '+str(mse))
            mae = mean_absolute_error(y_future, predictions["treePrediction"])
            print('MAE: '+str(mae))
            rmse = math.sqrt(mean_squared_error(y_future, predictions["treePrediction"]))
            print('RMSE: '+str(rmse))
            mape = np.mean(np.abs(predictions["treePrediction"] - y_future)/np.abs(y_future))
            print('MAPE: '+str(mape))
            
            mets = {"MAE" : str(mae),
                    "MSE" : str(mse),
                    "RMSE": str(rmse),
                    "MAPE": str(mape)}
        
        else:
            mets = {}
        
        endData = pd.concat([predictions, priceData], axis = 1)
        plt.pyplot.close("all")
        
        return tree, endData, mets
        
    
    
    
    
    def linearRegression(self, ticker = "", 
                         savePlt = False, 
                         evaluate = True, 
                         predLen = 100, 
                         plotWindow = 600,
                         timestampstr = ""):
        # https://www.kaggle.com/code/rishidamarla/stock-market-prediction-using-decision-tree/notebook
        loadedData, t = self.analysis.loadFromDB(tickerList = [ticker],
                                                indicators = ["MA20", "OBV", "IDEAL"],
                                                extras = ["HIGH", "LOW", "ADJRATIO", "VOLUME",
                                                          "IDEAL_HIGH", "IDEAL_LOW", "IDEAL_TRIG"])
        
        priceData = loadedData[loadedData["ticker_symbol"] == ticker]
        priceData = priceData.drop(["ticker_symbol"], 1)
        priceData = priceData.drop(["recordDate"], 1)
        
        priceData["Prediction"] = priceData["adj_close"].shift(-predLen)
        
        X = np.array(priceData.drop(["Prediction"], 1))[:-predLen]
        y = np.array(priceData["Prediction"])[:-predLen]
        
        if evaluate:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
            
            x_future = priceData.drop(["Prediction"], 1)[:-predLen]
            x_future = x_future.tail(predLen)
            predIndex = max(x_future.index)
            x_future = np.array(x_future)
            
            y_future = priceData["Prediction"][:-predLen]
            y_future = y_future.tail(predLen)
            y_future = np.array(y_future)
            
        else:
            x_train = X
            y_train = y
            
            x_future = priceData.drop(["Prediction"], 1)
            x_future = x_future.tail(predLen*2)
            predIndex = max(x_future.index) - predLen
            x_future = np.array(x_future)
        
        
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        
        predictions = lr.predict(x_future)
        predictions = pd.DataFrame(predictions)
        predictions = pd.concat([predictions, pd.Series(list(range(predIndex, predIndex + len(predictions))))], axis = 1)
        predictions.columns = ["val","ind"]
        predictions = predictions.set_index("ind")
        
        valid = priceData[X.shape[0]:]
        
        plt.figure.Figure(figsize=(16,8))
        plt.pyplot.title("Model")
        plt.pyplot.xlabel("Days")
        plt.pyplot.ylabel("Close Price USD ($)")
        plt.pyplot.plot(priceData["adj_close"])
        plt.pyplot.plot(valid["adj_close"])
        plt.pyplot.plot(predictions)
        plt.pyplot.xlim([len(priceData)-plotWindow, len(priceData)+predLen])
        plt.pyplot.legend(["Original", "True", "Predicted"])
        
        if savePlt:
            plt.pyplot.savefig("./static/Linear_1_" + timestampstr + ".png")
        else:
            plt.pyplot.show()
        
        if evaluate:
            print("Linear Regression performance: ")
            mse = mean_squared_error(y_future, predictions)
            print('MSE: '+str(mse))
            mae = mean_absolute_error(y_future, predictions)
            print('MAE: '+str(mae))
            rmse = math.sqrt(mean_squared_error(y_future, predictions))
            print('RMSE: '+str(rmse))
            mape = np.mean(np.abs(predictions["val"] - y_future)/np.abs(y_future))
            print('MAPE: '+str(mape))
            
            mets = {"MAE" : str(mae),
                    "MSE" : str(mse),
                    "RMSE": str(rmse),
                    "MAPE": str(mape)}
        
        else:
            mets = {}
        
        endData = pd.concat([predictions, priceData], axis = 1)
        plt.pyplot.close("all")
        
        return lr, endData, mets
    
    
    
    
    def ARIMA(self, ticker = "", 
              confInterval = 0.2, 
              savePlt = False, 
              evaluate = True, 
              predLen = 500, 
              plotWindow=600,
              timestampstr = ""):
        
        # https://www.analyticsvidhya.com/blog/2021/07/stock-market-forecasting-using-time-series-analysis-with-arima-model/
        priceData, t = self.analysis.loadFromDB(tickerList = [ticker],
                                                indicators = ["MA20", "OBV", "IDEAL"],
                                                extras = ["HIGH", "LOW", "ADJRATIO", "VOLUME",
                                                          "IDEAL_HIGH", "IDEAL_LOW", "IDEAL_TRIG"])
                
        priceData = priceData[priceData["ticker_symbol"] == ticker]
        
        rcParams['figure.figsize'] = 10, 6
        df_log = np.log(priceData["adj_close"])
        
        if evaluate:
            train_data, test_data = df_log[3:int(len(df_log)*0.8)], df_log[int(len(df_log)*0.8):]
        else:
            train_data = df_log[3:int(len(df_log))]
            test_data = pd.concat([df_log, pd.Series([np.nan]*predLen)], ignore_index=True)
        
        
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
        if savePlt:
            plt.pyplot.savefig("./static/ARIMA_1_" + timestampstr + ".png")
        else:
            plt.pyplot.show()
        
        
        model = ARIMA(train_data, order=(1,1,2))
        fitted = model.fit()
        print(fitted.summary())
        
        
        preRes = fitted.get_forecast(steps = predLen)
        
        fc = np.exp(preRes.predicted_mean)
        conf = np.exp(preRes.conf_int(alpha=confInterval))
        train_data = np.exp(train_data)
        test_data = np.exp(test_data)
        
        # Make as pandas series
        fc_series = pd.Series(fc, index=test_data.index)
        lower_series = pd.Series(conf["lower adj_close"], index=test_data.index)
        upper_series = pd.Series(conf["upper adj_close"], index=test_data.index)
        # Plot
        plt.pyplot.figure(figsize=(10,5), dpi=100)
        plt.pyplot.plot(train_data, label='training data')
        plt.pyplot.plot(test_data, color = 'blue', label='Actual Stock Price')
        plt.pyplot.plot(fc_series, color = 'orange',label='Predicted Stock Price')
        plt.pyplot.fill_between(lower_series.index, lower_series, upper_series, 
                          color='k', alpha=.10)
        plt.pyplot.title(ticker)
        plt.pyplot.xlabel('Time')
        plt.pyplot.ylabel(ticker  + ' Stock Price')
        plt.pyplot.legend(loc='upper left', fontsize=8)
        plt.pyplot.xlim([len(priceData)-plotWindow, len(priceData)+predLen])
        if savePlt:
            plt.pyplot.savefig("./static/ARIMA_2_" + timestampstr + ".png")
        else:
            plt.pyplot.show()
        
        if evaluate:
            mse = mean_squared_error(test_data, fc)
            print('MSE: '+str(mse))
            mae = mean_absolute_error(test_data, fc)
            print('MAE: '+str(mae))
            rmse = math.sqrt(mean_squared_error(test_data, fc))
            print('RMSE: '+str(rmse))
            mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
            print('MAPE: '+str(mape))
        
            metrics = {"MAE": str(mae),
                       "MSE": str(mse),
                       "RMSE": str(rmse),
                       "MAPE": str(mape),
                       "Summary": str(fitted.summary())}
        
        else:
            metrics = {}
            
        plt.pyplot.close("all")
        
        return model, fitted, fc, conf, metrics, priceData
    
    
    
    
    def autoARIMA(self, ticker = "", 
                  confInterval = 0.2, 
                  savePlt = False, 
                  evaluate = True, 
                  predLen = 500, 
                  plotWindow = 600,
                  loadFromSave = True,
                  timestampstr = ""):
        
        # https://www.analyticsvidhya.com/blog/2021/07/stock-market-forecasting-using-time-series-analysis-with-arima-model/
        priceData, t = self.analysis.loadFromDB(tickerList = [ticker],
                                                indicators = ["MA20"],
                                                extras = [])
                
        priceData = priceData[priceData["ticker_symbol"] == ticker]
        
        rcParams['figure.figsize'] = 10, 6
        df_log = np.log(priceData["adj_close"])
        
        if evaluate:
            train_data, test_data = df_log[0:int(len(df_log)*0.8)], df_log[int(len(df_log)*0.8):]
        else:
            train_data = df_log[0:int(len(df_log))]
            test_data = pd.concat([df_log, pd.Series([np.nan]*predLen)], ignore_index=True)
        
        
        
        try:
            if loadFromSave:
                fileName = "./static/autoARIMA_models/" + ticker + ".pkl"
                with open(fileName, 'rb') as pkl:
                    model_autoARIMA = pickle.load(pkl)
            else:
                raise ValueError
                
        except:
            model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                              test='adf',         # use adftest to find optimal 'd'
                              max_p=10, max_q=10, # maximum p and q
                              m=1,                # frequency of series
                              d=None,             # let model determine 'd'
                              seasonal=False,     # No Seasonality
                              start_P=0, 
                              D=0, 
                              trace=True,
                              error_action='ignore',  
                              suppress_warnings=True, 
                              stepwise=True)
            
            fileName = "./static/autoARIMA_models/" + ticker + ".pkl"
            with open(fileName, 'wb') as pkl:
                pickle.dump(model_autoARIMA, pkl)
        
        
        print(model_autoARIMA.summary())
        model_autoARIMA.plot_diagnostics(figsize=(15,8))
        if savePlt:
            plt.pyplot.savefig("./static/ARIMA_1_" + timestampstr + ".png")
        else:
            plt.pyplot.show()
        
        fitted = model_autoARIMA.fit(train_data)
        
        fc, conf = fitted.predict(n_periods=predLen, return_conf_int=True, alpha=confInterval)
        
        fc = np.exp(fc)
        conf = np.exp(conf)
        
        train_data = np.exp(train_data)
        test_data = np.exp(test_data)
        
        # Make as pandas series
        fc_series = pd.Series(fc)
        lower_series = pd.Series(conf[:,0])
        upper_series = pd.Series(conf[:,1])
        
        indSeries = pd.Series(list(range(len(train_data),len(train_data) + predLen)))
        
        fc_series = pd.concat([fc_series, indSeries], axis = 1, ignore_index=True)
        lower_series = pd.concat([lower_series, indSeries], axis = 1, ignore_index=True)
        upper_series = pd.concat([upper_series, indSeries], axis = 1, ignore_index=True)
        
        fc_series.columns = ["mean_pred", "ind"]
        lower_series.columns = ["lower_conf", "ind"]
        upper_series.columns = ["upper_conf", "ind"]
        
        fc_series.set_index(["ind"], inplace = True)
        lower_series.set_index(["ind"], inplace = True)
        upper_series.set_index(["ind"], inplace = True)
        
        # Plot
        plt.pyplot.figure(figsize=(10,5), dpi=100)
        plt.pyplot.plot(train_data, label='training data')
        plt.pyplot.plot(test_data, color = 'blue', label='Actual Stock Price')
        plt.pyplot.plot(fc_series, color = 'orange',label='Predicted Stock Price')
        plt.pyplot.fill_between(lower_series.index, lower_series["lower_conf"], upper_series["upper_conf"], 
                          color='k', alpha=.10)
        plt.pyplot.title(ticker)
        plt.pyplot.xlabel('Time')
        plt.pyplot.ylabel(ticker  + ' Stock Price')
        plt.pyplot.legend(loc='upper left', fontsize=8)
        plt.pyplot.xlim([len(priceData)-plotWindow, len(priceData) + predLen])
        if savePlt:
            plt.pyplot.savefig("./static/ARIMA_2_" + timestampstr + ".png")
        else:
            plt.pyplot.show()
        
        if evaluate:
            mse = mean_squared_error(test_data, fc)
            print('MSE: '+str(mse))
            mae = mean_absolute_error(test_data, fc)
            print('MAE: '+str(mae))
            rmse = math.sqrt(mean_squared_error(test_data, fc))
            print('RMSE: '+str(rmse))
            mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
            print('MAPE: '+str(mape))
        
            metrics = {"MAE": str(mae),
                       "MSE": str(mse),
                       "RMSE": str(rmse),
                       "MAPE": str(mape),
                       "Summary": str(fitted.summary())}
        
        else:
            metrics = {}
            
        endData = pd.concat([fc_series["mean_pred"], lower_series["lower_conf"], 
                             upper_series["upper_conf"], priceData["adj_close"]], axis = 1)
        plt.pyplot.close("all")
        
        return model_autoARIMA, fitted, endData, metrics
    
        
        
        
        



    
if __name__ == "__main__":
    
    if 'mod' not in locals():
        mod = MLmodels()
        print("\n---- New instance of MLmodels created. ----")
    
    mod.analysis.filterStocksFromDataBase(dailyLength = 1250, 
                                          maxDailyChange = 50, 
                                          minDailyChange = -50, 
                                          minDailyVolume = 5000000)
    
    
    # mod.LSTM_train(EpochsPerTicker = 20, fullItterations = 10, loadPrevious = False, look_back = 250)
    # data = mod.LSTM_load()
    # prediction, evaluation, testX, [testYr, testYc] = mod.LSTM_eval(ticker = "TSLA", evaluate = False)
    # lstm_pred = mod.LSTM_test()
    
    # tree, endData, mets = mod.Trees("A", savePlt=True, evaluate = False)
    # lr_model = mod.linearRegression("A")
    # model_autoARIMA, fitted, endData = mod.autoARIMA("TSLA", evaluate=False, predLen=100, loadFromSave = False)
    
    
    
    
    
    
    
    