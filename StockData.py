# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:54:23 2021

getData.checkForErrorsUpdateTickerSymbol() has a bad SQL fuction with the use of 
'tickerErrAttr' and getData.saveToSQL in the cmd_string with 'table_name' as a 
function argument as the column value to call in the SQL request.  
Need better option...

@author: sreit
"""

import pandas as pd
import datetime
import time
import requests
import sqlite3
import os
import sys
import commonUtilities
import numpy as np


class callLimitExceeded(Exception):
    # raised when the program detects that the call limit was exceeded, 
    # either because the limit set in 'getData' is exceeded or if the API 
    # returns a specific error.
    pass


class getAlphaVantageData:
    def __init__(self, API_KEY = "NYLUQRD89OVSIL3I", rate = 5, limit = 500, dataBaseSaveFile = "./stockData.db", alphaPremiumKey = False, dataBaseThreadCheck = True):
        self.workingFileText = "" # string for printing data to the command line between function calls
        self.DB = database(dataBaseSaveFile, dataBaseThreadCheck = dataBaseThreadCheck) # SQL object
        self._cur = self.DB.stockDB.cursor() # Cursor for SQL transactions
        self._rate = rate   # calls per minute
        self._limit = limit  # calls per day
        self._apiTotalCalls = 0  # number of calls already made
        self._apiTotalCount = 0
        self._API_KEY = API_KEY  # Key for use with alphaVantage API
        self._alphaPremium = alphaPremiumKey
        self._alphaVantageBaseURL = "https://www.alphavantage.co/query?" # base URL for alphavantage
        apiCallTime = time.time() - 60 # one minute in the past; ensures that there is no delay prior to the first API calls
        
        # array of last several API calls within last minute; keeps track of when 
        # the API calls were made so that the per minute API rate limits are not exceeded
        self._apiRecentCallTimes = [apiCallTime for i in range(self._rate)] 
        
        # Dictionaries that converts a user-input value to the table/columns that 
        # should be accessed within the SQL database.  Helps to keep the 
        # database from being corrupted by user inputs.  First dict cooresponds
        # to the tables that align with the function calls.  The second is 
        # used to modify the values from the first dict to match the columns in 
        # the table 'ticker_symbol_list' (that holds a status board of sorts)
        self._tableConversion = commonUtilities.conversionTables.tickerConversionTable
        self._tickerListConversion = {"MOSTRECENT"  :  "date_",
                                      "ERRORCODE"   :  "error_",
                                      "NUMRECORDS"  :  "records_"}
        
        self.validate = commonUtilities.validationFunctions()
    
    
    
    def _resetTotalApiCalls(self):
        # function call to reset the API calls so that this quantity is not set directly
        self._apiTotalCalls = 0
        return 0
    
    
        
    def changeAPIKey(self, newKey):
        # takes a User input API key and assigns it to the API key variable
        self.validate.validateString(newKey)
        
        self._API_KEY = "".join(e for e in newKey if e.isalnum())
        
        print("New API key is:  '" + self._API_KEY + "'\n\n")
        return 0
    
    
    
    def changeLimit(self, newLimit):
        # Changes the number of daily calls to a user-assigned value
        self.validate.validateInteger(newLimit)
        
        self._limit = newLimit
        return 0
    
    
    
    def changeRate(self, newRate):
        # Changes the number of calls per minute to a user-assigned value
        self.validate.validateInteger(newRate)
            
        if(self._rate == newRate):
            return 0
        
        # Adjusts the list of call times to match the new rate.
        while len(self._apiRecentCallTimes) > newRate:
            self._apiRecentCallTimes.pop(-1)
        
        callMaxTime = max(self._apiRecentCallTimes)
        while len(self._apiRecentCallTimes) < newRate:
            self._apiRecentCallTimes.append(callMaxTime)
        
        self._rate = newRate
        
        return 0
    
    
    
    def _checkApiCalls(self):
        # function checks the per minute and daily call limits, raising an exception 
        # if the daily limit is exceeded or delaying execution cycles to match the 
        # per minute limits.
        self._apiTotalCalls += 1
        
        if self._apiTotalCalls > self._limit:
            raise callLimitExceeded("Daily API call limit met.  Please wait or increase limit.")
        
        # Pause execution for the per-minute limit
        while time.time() - self._apiRecentCallTimes[0] < 60:
            timeRemaining = int(self._apiRecentCallTimes[0] + 60 - time.time())
            print("\rDelay for API call rate.  Time remaining = " + str(timeRemaining) + "            ", end = "\r")
            time.sleep(1)
        
        # remove the first (oldest) element from the list and add the current time
        # to the end of the API call list.  
        self._apiRecentCallTimes.pop(0)
        self._apiRecentCallTimes.append(time.time())
        
        print("\rNumber of API calls today = " + str(self._apiTotalCalls).rjust(5) + ".  " + str(self._apiTotalCount - self._apiTotalCalls).rjust(6) + "  calls remaining.           ")
        
    
    
    def _checkForErrorsUpdateTickerSymbol(self, ticker_symbol, jsonResponse, funcName = ""):
        # look for errors in the JSON response from the API request
        # get the keys from the JSON response
        jsonKeys = list(jsonResponse.keys())
        
        if jsonKeys == []:
            msgType = "Empty"
            message = "API returned no information."
        else:
            msgType = jsonKeys[0]
            message = jsonResponse[msgType]
        
        # check to see if the API responds that the rate or limit were exceeded,
        # and raises an error.
        if "standard API call frequency is 5 calls per minute and 500 calls per day" in message:
            raise callLimitExceeded("API calls exhausted for today, or calls are too frequent.")
        
        # checks to see if there is a different issue (i.e. invalid call)
        if msgType in ["Error Message", "Note", "Information", "Empty"]:
            
            # add a line to the 'api_transactions' table indicating the specific error
            self._updateAPIStatus(ticker_symbol = ticker_symbol,
                                  callTime = str(time.time()),
                                  msgType = msgType,
                                  source = funcName, 
                                  message = message, 
                                  status = "Fail")
            
            # modify the 'ticker_symbol_list' table to indicate the error status
            if funcName != "":
                self._updateTickerList(ticker_symbol = ticker_symbol,  
                                       recordDate = str(datetime.date.today()),
                                       columnType = "ERRORCODE", 
                                       columnFunction = funcName,
                                       value = "1")
                
                self._updateTickerList(ticker_symbol = ticker_symbol,  
                                       recordDate = str(datetime.date.today()),
                                       columnType = "MOSTRECENT", 
                                       columnFunction = funcName,
                                       value = str(datetime.date.today()))
            
            # commit the changes
            self.DB.stockDB.commit()
            
            print("\r  Error on ticker '" + ticker_symbol + "' and function '" + funcName + "':  " + message)
            
            # return 1 if there was an error detected
            return 1
        
        # return 0 if there was no error
        return 0
    
    
    
    def _updateAPIStatus (self, 
                          ticker_symbol, 
                          callTime = str(time.time()), #  <-- might not actually call this function on default conditions.  Add time.time() to the function call to ensure it works
                          callDate = str(datetime.date.today()), 
                          msgType = None, 
                          source = None, 
                          message = None, 
                          status = None):
        
        # function that runs an update for the 'api_transactions' table.  
        # ticker_symbol should be a string of letters and potentially hyphens
        # callTime is the UTC time of the API call as a string; measured in seconds since 1 Jan 1970
        # callDate is a string representing the date of the call (YYYY-MM-DD format)
        # msgType is type of error (error, information, empty, etc) or "Success" if the call was successful
        # source is a string representing the function that generated the API call.  Translated through self._tableConversion()
        # message is the string containing the error details
        # status contains "Success" or "Fail" for whether the call passed or not
        
        # check inputs for problematic data
        self.validate.validateString(ticker_symbol)
        self.validate.validateString(callTime)
        self.validate.validateDateString(callDate)
        if source not in self._tableConversion.keys():
            raise ValueError("Source not in conversion table.")
            
        
        # insert a new record with the ticker and call time to the 'api_transactions' table
        ticker_symbol = "".join(e.upper() for e in ticker_symbol if (e.isalpha() or e == "-"))
        sqlString  = "INSERT OR IGNORE INTO api_transactions (ticker_symbol, call_time) \n"
        sqlString += "VALUES(?, ?)"
        argList = (ticker_symbol, callTime)
        self._cur.execute(sqlString, argList)
        
        
        # Create a query string and list of arguments to update the record made above
        # based on the other values passed to the function.  Each input is scrubbed 
        # for non-applicable characters.
        argList = []
        sqlString  = "UPDATE api_transactions \n SET "
        if isinstance(callDate, str):
            callDate = "".join(e for e in callDate if (e.isalnum() or e == "-"))
            argList.append(callDate)
            sqlString += " call_date = ?,\n   "
        if isinstance(msgType, str):
            msgType = "".join(e for e in msgType if (e.isalnum() or e in ["-", " "]))
            argList.append(msgType)
            sqlString += " type = ?,\n   "
        if isinstance(source, str):
            source = "".join(e for e in source if (e.isalnum() or e == "_"))
            argList.append(self._tableConversion[source].replace("_", " ").title())
            sqlString += " source = ?,\n   "
        if isinstance(message, str):
            message = "".join(e for e in message if (e.isalnum() or e in ["-", "_", ",", ".", " ", "/", "(", ")"]))
            argList.append(message)
            sqlString += " message = ?,\n   "
        if isinstance(status, str):
            status = "".join(e for e in status if e.isalpha())
            argList.append(status)
            sqlString += " status = ?,\n   "
        
        # finish the string, execute the SQL transaction, and commit the changes
        sqlString  = sqlString[:-5] + "\n"
        sqlString += "WHERE ticker_symbol = ? AND call_time = ?; \n"
        argList.append(ticker_symbol)
        argList.append(callTime)
        argList = tuple(argList)
        self._cur.execute(sqlString, argList)
        self.DB.stockDB.commit()
        
    
    
    def _updateTickerList(self, 
                          ticker_symbol, 
                          recordDate = str(datetime.date.today()), 
                          name = None, 
                          exchange = None, 
                          columnType = None, 
                          columnFunction = None, 
                          value = None):
        
        # function that runs an update for the 'ticker_symbol_list' table.  
        # ticker_symbol should be a string of letters and potentially hyphens
        # recordDate is a string representing the date of the call (YYYY-MM-DD format)
        # name is string for the company name
        # exchange is a string for the exchange the stock is traded on
        # columnType is a string key for self._tickerListConversion (values are 'date_', 'error_', and 'records_')
        # columnFunction is a string key for self._tableConversion (values are the tables/functions)
        # value is a string for the data that should be saved into the table
        
        # check inputs for problematic data
        self.validate.validateString(ticker_symbol)
        self.validate.validateDateString(recordDate)
        if name == None and exchange == None and value == None:
            raise ValueError("Name, Exchange,  *AND*  Value in 'updateTickerList()' are empty; nothing to update.  ")
        
        ticker_symbol = "".join(e.upper() for e in ticker_symbol if (e.isalpha() or e == "-"))
        columnType = "".join(e for e in columnType if e.isalpha())
        columnFunction = "".join(e for e in columnFunction if e.isalpha())
        
        columnType = self._tickerListConversion[columnType]
        columnFunction = self._tableConversion[columnFunction]
        columnName = columnType + columnFunction
        
        
        # Create a query string and list of arguments to update the record made above
        # based on the other values passed to the function.  Each input is scrubbed 
        # for non-applicable characters.
        argList = []
        
        sqlString  = "UPDATE ticker_symbol_list \n"
        sqlString += "SET "
        if isinstance(recordDate, str):
            recordDate = "".join(e.upper() for e in recordDate if (e.isalnum()) or e == "-")
            argList.append(recordDate)
            sqlString += " recordDate = ?,\n    "
        if isinstance(name, str):
            name = "".join(e for e in name if e.isalpha())
            argList.append(name)
            sqlString += " name = ?,\n    "
        if isinstance(exchange, str):
            exchange = "".join(e for e in exchange if e.isalpha())
            argList.append(exchange)
            sqlString += " exchange = ?,\n    "
        if isinstance(value, str):
            value = "".join(e.upper() for e in value if (e.isalnum() or e in ["-", "_", ",", ".", " "]))
            argList.append(value)
            sqlString += " " + columnName + " = ?,\n    "
        
        # finish the string, execute the SQL transaction, and commit the changes
        sqlString  = sqlString[:-6] + "\n"
        sqlString += "WHERE ticker_symbol = ?; \n"
        argList.append(ticker_symbol)
        argList = tuple(argList)
        
        self._cur.execute(sqlString, argList)
        self.DB.stockDB.commit()
    
    
    
    def _getTimeSeriesDailyAdjusted(self, ticker_symbol, save = True):
        # Collects the time Series data for a selected ticker
        
        if self._alphaPremium:
            stockDF = self._getDailyAdjusted(ticker_symbol = ticker_symbol)
        else:
            out1 = self._getDaily(ticker_symbol = ticker_symbol)
            out2 = self._getWeeklyAdjusted(ticker_symbol = ticker_symbol)
            
            stockDF = pd.concat([out1, out2], axis=1)
            stockDF.sort_index(inplace = True, ascending = False)
            
            stockDF["adjustment_ratio"] = stockDF["adjustment_ratio"].ffill()
            stockDF["adj_close"] = [float(c)/float(ar) for c,ar in zip(stockDF["close"], stockDF["adjustment_ratio"])]
            
            stockDF["open"]     = [float(o) for o in stockDF["open"]]
            stockDF["close"]    = [float(c) for c in stockDF["close"]]
            stockDF["low"]      = [float(l) for l in stockDF["low"]]
            stockDF["high"]     = [float(h) for h in stockDF["high"]]
            stockDF["volume"]   = [float(v) for v in stockDF["volume"]]
            stockDF["dividend"] = [float(d) for d in stockDF["dividend"]]
            
            stockDF.drop(columns = ["adjustment_ratio"], inplace = True)
            
        
        # optionally save the data to the SQLite database
        if save:
            self.saveToSQL(stockDF, "alphatime", ticker_symbol)
        
        return stockDF

    
    
    
    def _getDailyAdjusted(self, ticker_symbol):
        # Build the request URL for alphaVantage API; TIME_SERIES_DAILY_ADJUSTED now requires a subscription
        requestURL = self._alphaVantageBaseURL + "function=TIME_SERIES_DAILY_ADJUSTED&" + \
                     "outputsize=full&symbol=" + ticker_symbol + "&apikey=" + self._API_KEY
        
        # Check to make sure the API limit and rate set in __init__() have not 
        # been exceeded; function will delay execution to ensure that the rate 
        # is not exceeded.
        self._checkApiCalls()
        
        # Send API request and extract the JSON data
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self._checkForErrorsUpdateTickerSymbol(ticker_symbol, data, "alphatime"):
            return None
        
        # Extract the data from the response and convert it to a dataframe
        stockDF = pd.DataFrame(data["Time Series (Daily)"])
        stockDF = stockDF.transpose()
        
        stockDF.rename({"1. open": "open",
                        "2. high": "high",
                        "3. low": "low",
                        "4. close": "close",
                        "5. adjusted close": "adj_close",
                        "6. volume": "volume",
                        "7. dividend amount": "dividend",
                        "8. split coefficient": "split"}, axis=1, inplace=True)
        
        
        stockDF['ticker_symbol'] = ticker_symbol
        stockDF['recordDate'] = stockDF.index
        
        # return the downloaded data back from the function
        return stockDF
    
    
    
    
    def _getDaily(self, ticker_symbol):
        # Collects the time Series data for a selected ticker
                    
        # Build the request URL for alphaVantage API; TIME_SERIES_DAILY_ADJUSTED now requires a subscription
        requestURL = self._alphaVantageBaseURL + "function=TIME_SERIES_DAILY&" + \
                     "outputsize=full&symbol=" + ticker_symbol + "&apikey=" + self._API_KEY
        
        # Check to make sure the API limit and rate set in __init__() have not 
        # been exceeded; function will delay execution to ensure that the rate 
        # is not exceeded.
        self._checkApiCalls()
        
        # Send API request and extract the JSON data
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self._checkForErrorsUpdateTickerSymbol(ticker_symbol, data, "alphatime"):
            return None
        
        # Extract the data from the response and convert it to a dataframe
        stockDF = pd.DataFrame(data["Time Series (Daily)"])
        stockDF = stockDF.transpose()
        
        stockDF.rename({"1. open": "open",
                        "2. high": "high",
                        "3. low": "low",
                        "4. close": "close",
                        "5. volume": "volume"}, axis=1, inplace=True)
        
        stockDF['ticker_symbol'] = ticker_symbol
        stockDF['recordDate'] = stockDF.index
        
        # return the downloaded data back from the function
        return stockDF
    
    
    
    
    def _getWeeklyAdjusted(self, ticker_symbol):
        # Collects the time Series data for a selected ticker
                    
        # Build the request URL for alphaVantage API; TIME_SERIES_DAILY_ADJUSTED now requires a subscription
        requestURL = self._alphaVantageBaseURL + "function=TIME_SERIES_WEEKLY_ADJUSTED&" + \
                     "outputsize=full&symbol=" + ticker_symbol + "&apikey=" + self._API_KEY
        
        # Check to make sure the API limit and rate set in __init__() have not 
        # been exceeded; function will delay execution to ensure that the rate 
        # is not exceeded.
        self._checkApiCalls()
        
        # Send API request and extract the JSON data
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self._checkForErrorsUpdateTickerSymbol(ticker_symbol, data, "alphatime"):
            return None
        
        # Extract the data from the response and convert it to a dataframe
        stockDF = pd.DataFrame(data["Weekly Adjusted Time Series"])
        stockDF = stockDF.transpose()
    
        stockDF["adjustment_ratio"] = [float(a)/float(b) for a,b in zip(stockDF["4. close"], stockDF["5. adjusted close"])]
        stockDF.rename({"5. adjusted close": "adj_close",
                        "7. dividend amount": "dividend"}, axis=1, inplace=True)
        stockDF["split"] = [np.nan] * len(stockDF["adj_close"])
        stockDF["recordDate"] = stockDF.index
        
        stockDF.drop(columns = ["1. open", "2. high", "3. low", "4. close", "6. volume", "recordDate"], inplace=True)
        
        # return the downloaded data back from the function
        return stockDF
    
    
    
    

    def _getFundamentalOverview(self, ticker_symbol, save = True):
        # Collects the company overview/fundamental data for a selected ticker
                    
        # Build the request URL for alphaVantage API
        requestURL = self._alphaVantageBaseURL + "function=OVERVIEW&" + \
                     "symbol=" + ticker_symbol + "&apikey=" + self._API_KEY
        
        # Check to make sure the API limit and rate set in __init__() have not 
        # been exceeded; function will delay execution to ensure that the rate 
        # is not exceeded.
        self._checkApiCalls()
        
        # Send API request and extract the JSON data
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self._checkForErrorsUpdateTickerSymbol(ticker_symbol, data, "overview"):
            return None 
        
        # Extract the data from the response and convert it to a dataframe
        stockDF = pd.DataFrame(list(data.items()), index = data.keys(), columns = ["Label","Data"])
        
        stockDF.rename(index={"Symbol":"ticker_symbol", \
                              "52WeekHigh":"high_52week", \
                              "52WeekLow":"low_52week", \
                              "50DayMovingAverage":"moving_average_50d", \
                              "200DayMovingAverage":"moving_average_200d"}, inplace=True)
            
        # transpose the dataframe and add a marker for the date the data was saved
        stockDF = stockDF.transpose()
        stockDF.drop('Label', inplace = True)
        
        stockDF['recordDate'] = str(datetime.date.today())
        
        # optionally save the data to the SQLite database
        if save:
            self.saveToSQL(stockDF, "overview", ticker_symbol)
        
        # return the downloaded data back from the function
        return stockDF
    


    def _getCashFlow(self, ticker_symbol, save = True):
        # Collects the cash flow data for a selected ticker
                    
        # Build the request URL for alphaVantage API
        requestURL = self._alphaVantageBaseURL + "function=CASH_FLOW&" + \
                     "symbol=" + ticker_symbol + "&apikey=" + self._API_KEY
        
        # Check to make sure the API limit and rate set in __init__() have not 
        # been exceeded; function will delay execution to ensure that the rate 
        # is not exceeded.
        self._checkApiCalls()
        
        # Send API request and extract the JSON data
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self._checkForErrorsUpdateTickerSymbol(ticker_symbol, data, "cash"):
            return None 
        
        # separate the annual and quarterly data from the JSON and convert it 
        # to a dataframe
        annual = pd.DataFrame(data['annualReports'])
        quarterly = pd.DataFrame(data['quarterlyReports'])
        
        # Add boolean column for annual vs quarterly, and a column for the 
        # ticker symbol
        annual['annualReport'] = 1
        quarterly['annualReport'] = 0
        annual['ticker_symbol'] = ticker_symbol
        quarterly['ticker_symbol'] = ticker_symbol
        
        # concatenate the quarterly and annual dataframes
        frames = [annual, quarterly]
        stockDF = pd.concat(frames)
        
        stockDF.rename({"proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet":
                        "proceedsFromIssuanceOfLongTermDebt", \
                        "fiscalDateEnding":"recordDate"}, axis=1, inplace=True)
        
        # optionally save the data to the SQLite database
        if save:
            self.saveToSQL(stockDF, "cash", ticker_symbol)
            
        # return the downloaded data back from the function
        return stockDF
    
    
    
    def _getIncomeStatement(self, ticker_symbol, save = True):
        # Collects the income statement data for a selected ticker
                    
        # Build the request URL for alphaVantage API
        requestURL = self._alphaVantageBaseURL + "function=INCOME_STATEMENT&" + \
                     "symbol=" + ticker_symbol + "&apikey=" + self._API_KEY
        
        # Check to make sure the API limit and rate set in __init__() have not 
        # been exceeded; function will delay execution to ensure that the rate 
        # is not exceeded. 
        self._checkApiCalls()
        
        # Send API request and extract the JSON data
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self._checkForErrorsUpdateTickerSymbol(ticker_symbol, data, "income"):
            return None 
        
        # separate the annual and quarterly data from the JSON and convert it 
        # to a dataframe
        annual = pd.DataFrame(data['annualReports'])
        quarterly = pd.DataFrame(data['quarterlyReports'])
        
        # Add boolean column for annual vs quarterly, and a column for the 
        # ticker symbol
        annual['annualReport'] = 1
        quarterly['annualReport'] = 0
        annual['ticker_symbol'] = ticker_symbol
        quarterly['ticker_symbol'] = ticker_symbol
        
        # concatenate the quarterly and annual dataframes
        frames = [annual, quarterly]
        stockDF = pd.concat(frames)
        
        stockDF.rename({"fiscalDateEnding":"recordDate"}, axis=1, inplace=True)
        
        # optionally save the data to the SQLite database
        if save:
            self.saveToSQL(stockDF, "income", ticker_symbol)

        # return the downloaded data back from the function
        return stockDF



    def _getEarnings(self, ticker_symbol, save = True):
        # Collects the earnings data for a selected ticker
                    
        # Build the request URL for alphaVantage API
        requestURL = self._alphaVantageBaseURL + "function=EARNINGS&" + \
                     "symbol=" + ticker_symbol + "&apikey=" + self._API_KEY
        
        # Check to make sure the API limit and rate set in __init__() have not 
        # been exceeded; function will delay execution to ensure that the rate 
        # is not exceeded.  
        self._checkApiCalls()
        
        # Send API request and extract the JSON data
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self._checkForErrorsUpdateTickerSymbol(ticker_symbol, data, "earnings"):
            return None
        
        # separate the annual and quarterly data from the JSON and convert it 
        # to a dataframe
        stockDF = pd.DataFrame(data['quarterlyEarnings'])
        
        # Add a column for the ticker symbol
        stockDF['ticker_symbol'] = ticker_symbol
        
        stockDF.rename({"fiscalDateEnding":"recordDate"}, axis=1, inplace=True)
        
        # optionally save the data to the SQLite database
        if save:
            self.saveToSQL(stockDF, "earnings", ticker_symbol)

        # return the downloaded data back from the function
        return stockDF



    def _getBalanceSheet(self, ticker_symbol, save = True):
        # Collects the balance sheet data for a selected ticker
                    
        # Build the request URL for alphaVantage API
        requestURL = self._alphaVantageBaseURL + "function=BALANCE_SHEET&" + \
                     "symbol=" + ticker_symbol + "&apikey=" + self._API_KEY
        
        # Check to make sure the API limit and rate set in __init__() have not 
        # been exceeded; function will delay execution to ensure that the rate 
        # is not exceeded.  
        self._checkApiCalls()
        
        # Send API request and extract the JSON data
        response = requests.get(requestURL)
        data = response.json()
        
        # Check the response for errors
        if self._checkForErrorsUpdateTickerSymbol(ticker_symbol, data, "balance"):
            return None 
        
        # separate the annual and quarterly data from the JSON and convert it 
        # to a dataframe
        annual = pd.DataFrame(data['annualReports'])
        quarterly = pd.DataFrame(data['quarterlyReports'])
        
        # Add boolean column for annual vs quarterly, and a column for the 
        # ticker symbol
        annual['annualReport'] = 1
        quarterly['annualReport'] = 0
        annual['ticker_symbol'] = ticker_symbol
        quarterly['ticker_symbol'] = ticker_symbol
        
        # concatenate the quarterly and annual dataframes
        frames = [annual, quarterly]
        stockDF = pd.concat(frames)
        
        stockDF.rename({"fiscalDateEnding":"recordDate"}, axis=1, inplace=True)
        
        # optionally save the data to the SQLite database
        if save:
            self.saveToSQL(stockDF, "balance", ticker_symbol)
        
        # return the downloaded data back from the function
        return stockDF
    


    def saveToSQL(self, stockDF, table, ticker_symbol = ""):
        # Saves a dataframe to the SQLite database.  Functions as a wrapper for
        # this class and its dataframes.  The dataframes and SQLite schema were
        # setup to match and work well together.
        
        print(ticker_symbol + "     " + self._tableConversion[table].replace("_", " ").title() + "       ", end = "")
        
        # Check to see if the dataframe passed is actually empty, execute an
        # api_transaction for the empty dataframe, and move on to the next call.
        if stockDF.empty:
            self._emptyDF(stockDF, table, ticker_symbol)
            return 1
        
        # Create an entry for in the respective table for the API call that was made 
        if table in ["balance", "cash"]:
            insert_string  = "INSERT OR IGNORE INTO " + self._tableConversion[table]
            insert_string += " (ticker_symbol, recordDate, annualReport) "
            insert_string += "VALUES(?, ?, ?) \n"
            insert_list = stockDF[["ticker_symbol","recordDate","annualReport"]].values.tolist()
        else:
            insert_string  = "INSERT OR IGNORE INTO " + self._tableConversion[table]
            insert_string += " (ticker_symbol, recordDate) "
            insert_string += "VALUES(?, ?) \n"
            insert_list = stockDF[["ticker_symbol","recordDate"]].values.tolist()
                
        self._cur.executemany(insert_string, insert_list)
        self.DB.stockDB.commit()
        
        
        # Update the record created above with the other data passed to the function.
        update_string = "UPDATE " + self._tableConversion[table] + " SET \n"
        argList = pd.DataFrame()
        
        for key in stockDF.keys():
            if key not in ["ticker_symbol", "recordDate", "annualReport"]:
                update_string += "  " + key + " = ?,\n"
                argList[key] = stockDF[key]
        
        # Add 'where' clause to the SQL transaction to ensure that the correct record is altered.
        update_string = update_string[:-2] + "\nWHERE "
        if table in ["balance", "cash"]:
            # Close out the strings for the SQL transaction
            update_string += "ticker_symbol = ? \n"
            update_string += "AND recordDate = ? \n"
            update_string += "AND annualReport = ?; \n"
            argList["ticker_symbol"] = stockDF["ticker_symbol"]
            argList["recordDate"] = stockDF["recordDate"]
            argList["annualReport"] = stockDF["annualReport"]
        else:
            # Close out the strings for the SQL transaction
            update_string += "ticker_symbol = ? \n"
            update_string += "AND recordDate = ?; \n"
            argList["ticker_symbol"] = stockDF["ticker_symbol"]
            argList["recordDate"] = stockDF["recordDate"]
            
        argList = argList.values.tolist()
        
        # Execute the transactions and update the 'api_transactions' table with 
        # a success message, and the 'ticker_symbol_list' table with the 
        # respective information.
        self._cur.executemany(update_string, argList)
        
        self._updateTickerList(ticker_symbol = ticker_symbol,
                               columnType = "NUMRECORDS",
                               columnFunction = table,
                               value = str(len(stockDF["ticker_symbol"])))
        self._updateTickerList(ticker_symbol = ticker_symbol,
                               columnType = "ERRORCODE",
                               columnFunction = table,
                               value = "0")
        self._updateTickerList(ticker_symbol = ticker_symbol,
                               columnType = "MOSTRECENT",
                               columnFunction = table,
                               value = str(datetime.date.today()))
        
        self._updateAPIStatus(ticker_symbol = ticker_symbol, 
                              callTime = str(time.time()),
                              msgType = "Success",
                              source = table,
                              message = "Data saved successfully.",
                              status = "Success")
        
        print("\r" + self.workingFileText + ticker_symbol.rjust(6) + " records written: " + str(len(stockDF["ticker_symbol"])).rjust(7) + "  to table '" + self._tableConversion[table].replace("_", " ").title() + "'", end = "")
        
        print("\nDone", end = "")
    
    
    
    def _emptyDF(self, stockDF, table, ticker_symbol):
        # creates the entries when the API provides a non-empty responcse but 
        # the dataframe extracted from the response is empty.
        msgType = "Error"
        message = "API returned information; extracted DataFrame was empty."
        
        self._updateAPIStatus(ticker_symbol = ticker_symbol, 
                              callTime = str(time.time()),
                              msgType = msgType,
                              source = table,
                              message = message,
                              status = "Fail")
        
        self._updateTickerList(ticker_symbol = ticker_symbol,
                               columnType = "MOSTRECENT",
                               columnFunction = table,
                               value = str(datetime.date.today()))
        self._updateTickerList(ticker_symbol = ticker_symbol,
                               columnType = "NUMRECORDS",
                               columnFunction = table,
                               value = "0")
        self._updateTickerList(ticker_symbol = ticker_symbol,
                               columnType = "ERRORCODE",
                               columnFunction = table,
                               value = "1")
        
        print("\r  Error on ticker '" + ticker_symbol + "' and function '" + self._tableConversion[table].replace("_", " ").title() + "':  " + message)
    
    
    
    def addPickleToTickerList(self, directory = "./"):
        print("Adding entries for stocks in pickle files...")
        
        # Read the database and see what trackers are already in the database.
        # allows for the insertion of new records if the record doesn't exist
        query = self._cur.execute("SELECT * FROM ticker_symbol_list")
        cols = [column[0] for column in query.description]
        DF_tickerList = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
        DF_tickerList = list(DF_tickerList["ticker_symbol"])
        
        
        # Collect a list of file names in the directory holding the pickle files
        dirList = os.listdir(directory)
        
        DF_dirList = []
        
        for file in dirList:
            # Parse file names; files are named like this:
            #
            # 2_AAAA_Type_detailedType.pickle
            # keep the stock ticker ("AAAA") and append it to DF_driList
            
            fileName = file.split(".")[0].split("_")
            DF_dirList.append(fileName[1])
        
        # Remove all the repeat tickers (potentially from files containing
        # different data), remove those that are already in the SQLite database,
        # and then sort alphabetically.  
        DF_dirList = list(dict.fromkeys(DF_dirList))
        DF_dirList = list(set(DF_dirList) - set(DF_tickerList))
        DF_dirList.sort()
        
        # Convert the list of tickers from the directory to a pandas dataframe,
        # transpose it, and then add the date that the information was added to
        # to the SQLite database.
        DF_tickers = pd.DataFrame([DF_dirList])
        DF_tickers = DF_tickers.transpose()
        DF_tickers["recordDate"] = str(datetime.date.today())
        
        # add column headers and save to the database
        DF_tickers.columns = ["ticker_symbol", "recordDate"]
        self.saveToSQL(DF_tickers, "ticker_symbol_list")
         
        print("Done.")
        
        # return the list of tickers added for use outside the function
        return DF_tickers
    
    
    
    def copyPickleToSQL(self, stocksDirectory = "./"):
        # Process the pickle files generated from a previous script for 
        # inclusion into the SQLite database.  "Conversion" is a dictionary
        # translates the last part of the filename to the SQLite table
        # that the data needs to be entered into.
        
        pickleConversion = {"TimeData":"daily_adjusted",
                            "annualBalance":"balance_sheet",
                            "quarterlyBalance":"balance_sheet",
                            "annualCash":"cash_flow",
                            "quarterlyCash":"cash_flow",
                            "quarterlyEarnings":"earnings",
                            "Overview":"fundamental_overview",
                            "annualIncome":"income_statement",
                            "quarterlyIncome":"income_statement"}
        
        self.addPickleToTickerList(stocksDirectory)
        
        # Collect a list of file names in the directory holding the pickle files
        dirList = os.listdir(stocksDirectory)
        
        # Total and count for progress monitor
        numberOfFiles = str(len(dirList)).rjust(6)
        count = 0
        
        # Itterate through all the .pickle files to sort the data into the SQLite
        # tables and entries.  The Schema included in the Database class matches
        # the values in the .pickle files, and by extension the data downloaded
        # from the AlphaVantage API.
        for file in dirList:
            # Keep track of overall progress
            count += 1
            self.workingFileText = "Working file " + str(count).rjust(6) + " of " + numberOfFiles + "      "
            
            
            # Parse file names; files are named like this:
            #
            # 2_AAAA_Type_detailedType.pickle
            #
            # The name can be split along the underscores to give:
            # 2: ordinal number of the ticker; basically an inventory number
            # AAAA: Stock ticker as listed on its exchange
            # Type: which API call generated the data
            # detailedType: API call, plus frequency
            
            typeFile = file.split(".")[0].split("_")[-1]
            ticker_symbol = file.split(".")[0].split("_")[1]
            
            
            # Read the pickle file into a pandas dataframe.
            df = pd.read_pickle(stocksDirectory + file)
            
            # Series of if statements that determine the which type of data is 
            # in the dataframe, followed by a series of adustments that are used
            # to prepare the dataframe for entry into the SQLite database.
            # 
            # Most of this section renames pandas series and includes basic data
            # for each record (ticker symbol, date of the information, latest 
            # date that the information was pulled for the ticker symbol, etc.)
            if typeFile == "TimeData":
                df.rename({"adjusted close": "adj_close",
                           "dividend amount": "dividend",
                           "split coefficient": "split"}, axis=1, inplace=True)
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                tableName = "time"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "annualBalance":
                df['annualReport'] = 1
                df['ticker_symbol'] = ticker_symbol
                df["recordDate"] = df.index.date
                tableName = "balance"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "quarterlyBalance":
                df['annualReport'] = 0
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                tableName = "balance"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "annualCash":
                df['annualReport'] = 1
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                df.rename({"proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet":
                           "proceedsFromIssuanceOfLongTermDebt"}, axis=1, inplace=True)
                tableName = "cash"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "quarterlyCash":
                df['annualReport'] = 0
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                df.rename({"proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet":
                           "proceedsFromIssuanceOfLongTermDebt"}, axis=1, inplace=True)
                tableName = "cash"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "annualEarnings":
                continue
            
            elif typeFile == "quarterlyEarnings":
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                tableName = "earnings"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "Overview":
                df = pd.DataFrame(df)
                df["Label"] = df.index
                df.rename(index={"Symbol":"ticker_symbol", \
                                 "52WeekHigh":"high_52week", \
                                 "52WeekLow":"low_52week", \
                                 "50DayMovingAverage":"moving_average_50d", \
                                 "200DayMovingAverage":"moving_average_200d"}, inplace=True)
                df = df.transpose()
                df.drop('Label', inplace = True)
                df['recordDate'] = df["LatestQuarter"]
                tableName = "overview"
                infoDate = df["LatestQuarter"]
                
            elif typeFile == "annualIncome":
                df['annualReport'] = 1
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                tableName = "income"
                infoDate = str(df.index.max().date())
                
            elif typeFile == "quarterlyIncome":
                df['annualReport'] = 0
                df['ticker_symbol'] = ticker_symbol
                df['recordDate'] = df.index.date
                tableName = "income"
                infoDate = str(df.index.max().date())
            
            
            # Validates that the date associated with the most recent time the
            # data was current is actually formated as a date.
            try:
                self.validateDateString(infoDate)
            except:
                infoDate = "2021-08-06"
            
            # Name of the table that the data needs to be entered into, as
            # determined by the file name and the converstion table above.
            tableName = pickleConversion[typeFile]
            
            # for column in df:
            #     pd.to_numeric(df[column], errors = 'ignore')
            # return df
            
            # Save the data into the SQLite database.  Takes several minutes
            # for the series of files I had (~4.6 GB when complete)
            self.saveToSQL(df, tableName, ticker_symbol)
            
            # Update the ticker table data status; sets flags for whether there
            # were any errors downloading the data (no), whether the data in the
            # database is valid (yes), and when the data was last requested from 
            # the AlphaVantage API.  
            
            self._updateTickerList(ticker_symbol = ticker_symbol,
                                   recordDate = infoDate,
                                   columnType = "ERRORCODE",
                                   columnFunction = tableName,
                                   value = "0")
            
            self._updateTickerList(ticker_symbol = ticker_symbol,
                                   recordDate = infoDate,
                                   columnType = "MOSTRECENT",
                                   columnFunction = tableName,
                                   value = infoDate)
            
            self._updateTickerList(ticker_symbol = ticker_symbol,
                                   recordDate = infoDate,
                                   columnType = "NUMRECORDS",
                                   columnFunction = tableName,
                                   value = str(len(df.index)))
        
        self.workingFileText = ""
    
    
    
    def autoUpdate(self, 
                   metrics = None, 
                   stockList = None, 
                   startTicker = None, 
                   missing = True, 
                   error = False, 
                   olderThan = None, 
                   numRecords = None):
        
        # Automatically run through the database and identify missing, old, or 
        # erroroneous data for update, then get the data from AlphaVantage.
        # There are several conditions that are checked and used to build
        # a SQL query string that is ultimately run in the database.
                
        # check inputs to the function to ensure conformity
        if isinstance(stockList, str):
            stockList = ["".join(e for e in stockList if e.isalpha())]
            
        if isinstance(metrics, str):
            metrics = ["".join(e for e in metrics if e.isalpha())]
            
        if isinstance(startTicker, list):
            startTicker = startTicker[0]
            
        if isinstance(olderThan, str):
            try:
                self.validate.validateDateString(olderThan)
                then = datetime.datetime.strptime(olderThan, '%Y-%m-%d').date()
                now  = datetime.datetime.now().date()
                olderThan = (now - then).days
            except:
                raise ValueError("'olderThan' not recognized as a date (format: '%Y-%m-%d').")
                
        
        if not (isinstance(metrics, list) or metrics == None) or \
           not (isinstance(stockList, list) or stockList == None) or \
           not (isinstance(startTicker, str) or startTicker == None) or \
           not (isinstance(missing, bool)) or \
           not (isinstance(error, bool)) or \
           not (isinstance(olderThan, int) or olderThan == None) or \
           not (isinstance(numRecords, int) or numRecords == None):
            raise TypeError("At least one input to autoUpdate is not of the correct type./n")
        
        
        
        
        # default metrics to be all of the metrics; otherwise call the specific
        # functions for the data requested.
        if metrics == None:
            metrics = ["alphatime", "balance", "cash", "earnings", "overview", "income"]
            
        
        # Need a starting point for building the SQL query string
        # metricListString contains the stock filter conditions
        # queryStringBase contains the base request with parameters that get replaced 
        # before the SQL request is sent.
        
        # "WHERE NOT 1=1 " ensures that there is at least one condition after where
        # to prevent sytax errors at runtime.
        metricListString = ""
        queryStringBase  = "SELECT ticker_symbol " +\
                           "FROM ticker_symbol_list \n" +\
                           "WHERE "
        
        
        # Start the filtering string.  Start depends on whether the 
        if error or missing or olderThan != None:
            metricListString += " ("
        else: 
            metricListString += " (1=1 OR "
        
        
        # Add conditions that will evaluate to true if the previous request 
        # resulted in an error.
        if missing:
            metricListString += "date_[METRICS] IS NULL OR "
            metricListString += "error_[METRICS] = -1 OR "
        if error:
            metricListString += "error_[METRICS] = 1 OR "
        # Check to see if the data is too old.  Default is to ignore the date.
        # In SQLite, the date is saved as a string, so checking to see if the 
        # the stored date is alphabetically above or below the date requested
        # results in the correct filter.
        if olderThan != None:
            metricListString += "date_[METRICS] < '" + str(datetime.date.today()-datetime.timedelta(days = olderThan)) + "' OR "
        
        metricListString = metricListString[:-4] + ") \n"
        
        
        # filter the results to only include ticker symbols at or after the
        # user selected ticker
        if startTicker != None:
            startTicker = "".join(e for e in startTicker if e.isalpha()) # verify that all the text is alphabetical
            metricListString += "AND ticker_symbol >= '" + startTicker.upper() + "' \n"
            
        # filter by a list of specific user added stock tickers 
        if stockList != None:
            metricListString += "AND ("
            for stock in stockList:
                stock = "".join(e for e in stock if e.isalpha())
                metricListString += "ticker_symbol = '" + stock.upper() + "' OR "
            metricListString = metricListString[:-4] + ") \n"
        
        # dictionary that contains all the results from the search.  Each key 
        # is a ticker, and the value is a list of the metrics that need to be 
        # downloaded for that ticker.  This means that if a ticker only needs 
        # one metric, that is the only metric downloaded.  
        resultsDict = {}
        
        # Number of required API calls. can be used for tracking
        self._apiTotalCount = 0
        
        # Execute the search for each metric in the database and compile the results.
        for metric in metrics:
            queryString = queryStringBase + metricListString.replace("[METRICS]", self._tableConversion[metric])
            
            # Limit search to only the first 'numRecords' of records from the request 
            if numRecords == None:
                queryString = queryString[:-1] + ";\n"
            else:
                queryString += "LIMIT " + str(numRecords) + ";\n"
            
            
            # Execute the SQL request and convert it to a dataframe
            query = self._cur.execute(queryString)
            results = query.fetchall()
            
            # Combine the results into a dictionary of lists, where each key 
            # is a ticker, and each value is a list of metrics.
            for ticker in results:
                self._apiTotalCount += 1
                try:
                    resultsDict[ticker[0]].append(metric)
                except:
                    resultsDict[ticker[0]] = [metric]
                
        
        # Combine and sort the keys into a list, limiting the list of tickers 
        # to a length of 'numRecords'
        resultList = list(resultsDict.keys())
        resultList.sort()
        
        if numRecords != None:
            resultList = resultList[:numRecords]
            self._apiTotalCount = 0
            
            for stock in resultList:
                for metric in resultsDict[stock]:
                    self._apiTotalCount += 1
                
        # Download the selected metrics from AlphaVantage for the tickers 
        # returned from the SQL query.  
        for stock in resultList:
            for metric in resultsDict[stock]:
                if metric == "alphatime":
                    self._getTimeSeriesDaily(stock)
                elif metric == "balance":
                    self._getBalanceSheet(stock)
                elif metric == "cash":
                    self._getCashFlow(stock)
                elif metric == "earnings":
                    self._getEarnings(stock)
                elif metric == "overview":
                    self._getFundamentalOverview(stock)
                elif metric == "income":
                    self._getIncomeStatement(stock)
                self.DB.stockDB.commit()
        
        return 0



    def confirmDatesAndCounts(self, 
                              metrics = None, 
                              stockList = None, 
                              startTicker = None,
                              endTicker = None,
                              numRecords = None):
        
        # Automatically read through the SQL database and ensure that the 
        # 'ticker_symbol_list' table has accurate information (verify error codes,
        # dates, and amount of available data is accurate).
        
        # check function inputs for problematic values
        if isinstance(stockList, str):
            stockList = ["".join(e for e in stockList if (e.isalpha() or e == "-"))]
        if isinstance(metrics, str):
            metrics = ["".join(e for e in metrics if e.isalpha())]
        if isinstance(startTicker, list):
            startTicker = startTicker[0]
        if isinstance(startTicker, str):
            startTicker = ["".join(e.upper() for e in startTicker if (e.isalpha() or e == "-"))]
        if isinstance(endTicker, list):
            startTicker = startTicker[0]
        if isinstance(endTicker, str):
            endTicker = ["".join(e.upper() for e in endTicker if (e.isalpha() or e == "-"))]
        
        if not (isinstance(metrics, list) or metrics == None) or \
           not (isinstance(stockList, list) or stockList == None) or \
           not (isinstance(startTicker, str) or startTicker == None) or \
           not (isinstance(endTicker, str) or endTicker == None) or \
           not (isinstance(numRecords, int) or numRecords == None):
            raise TypeError("At least one input to confirmDatesAndCounts() is not of the correct type./n")
        
        
        # default metrics to be all of the metrics; otherwise call the specific
        # functions for the data requested.
        if metrics == None:
            metrics = list(self._tableConversion.keys())
        
        # "WHERE NOT 1=1 " ensures that there is at least one condition after where
        # to prevent sytax errors at runtime.
        queryString  = "SELECT ticker_symbol FROM ticker_symbol_list \n" +\
                       "WHERE 1=1 \n "
        
        # filter the results to only include ticker symbols at or after the
        # user selected ticker
        if startTicker != None:
            queryString += "AND ticker_symbol >= '" + startTicker + "' \n"
            
        # filter the results to only include ticker symbols before the
        # user selected ticker
        if endTicker != None:
            queryString += "AND ticker_symbol <= '" + endTicker + "' \n"
            
        # filter by a list of specific user added stock tickers 
        if stockList != None:
            queryString += "AND ("
            for stock in stockList:
                stock = "".join(e for e in stock if e.isalpha())
                queryString += "ticker_symbol = '" + stock.upper() + "' OR "
            queryString = queryString[:-4] + ") \n"
            
        # Limit search to only the first 'numRecords' of records from the request 
        if numRecords == None:
            queryString = queryString[:-1] + ";\n"
        else:
            queryString += "LIMIT " + str(numRecords) + ";\n"
        
        
        # Execute the SQL request and convert it to a dataframe and extract a 
        # list of the tickers that need to be referenced from the other tables.
        query = self._cur.execute(queryString)
        cols = [column[0] for column in query.description]
        results = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
        tickers = list(results["ticker_symbol"])
        
        # Execute the search for each metric in the database and compile the results.
        countProgress = 0
        for ticker in tickers:
            
            # construct the query string and argument list
            queryString  = "SELECT call_time, source, status "
            queryString += "FROM api_transactions \n"
            queryString += "WHERE ticker_symbol = ?"
            
            argList = [ticker]
            
            query = self._cur.execute(queryString, argList)
            cols = [column[0] for column in query.description]
            errors = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
            
            countProgress += 1
            print("\rWorking ticker:  " + ticker.rjust(7) + "  " + str(countProgress-1) + " of " + str(len(tickers)) + " complete", end = "")
            
            for metric in metrics:
                # get all the records within the table associated with a metric
                queryString  = "SELECT ticker_symbol, recordDate "
                queryString += "FROM " + self._tableConversion[metric] + " \n"
                queryString += "WHERE ticker_symbol = ?"
                
                argList = [ticker]
                
                # execute the SQL query and parse the response to a pandas dataframe
                query = self._cur.execute(queryString, argList)
                cols = [column[0] for column in query.description]
                results = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
                
                # count the number of records and get the most recent date of the records.
                # also set the error code to "0", representing successful download of data
                # if data exists, -1 if there is no record of data and no record of the 
                # ticker in the error list, or 1 if the ticker appears in the error list.
                if len(results["recordDate"]) > 0:
                    count = len(results["recordDate"])
                    recentDate = str(max(results["recordDate"]))
                
                    self._updateTickerList(ticker_symbol = ticker,
                                           columnType = "ERRORCODE",
                                           columnFunction = metric,
                                           value = "0")
                
                
                else:
                    count = 0
                    recentDate = ""
                    
                    resDF = errors[errors['source'] == self._tableConversion[metric].replace("_", " ").title()]
                    if resDF['call_time'].count() > 0:
                        self._updateTickerList(ticker_symbol = ticker,
                                               columnType = "ERRORCODE",
                                               columnFunction = metric,
                                               value = "1")
                        
                    else:
                        self._updateTickerList(ticker_symbol = ticker,
                                               columnType = "ERRORCODE",
                                               columnFunction = metric,
                                               value = "-1")
                
                
                self._updateTickerList(ticker_symbol = ticker,
                                       columnType = "NUMRECORDS",
                                       columnFunction = metric,
                                       value = str(count))
                
                self._updateTickerList(ticker_symbol = ticker,
                                       columnType = "MOSTRECENT",
                                       columnFunction = metric,
                                       value = recentDate)
                
                



class getYahooData:
    
    def __init__(self, API_KEY = "ie3hWXyolF4uBvD5V0A8t8gZ99GWStAS5soVxZl6", dataBaseSaveFile = "./SQLiteDB/stockData.db"):
        self.DB = database(dataBaseSaveFile) # SQL object
        self._cur = self.DB.stockDB.cursor() # Cursor for SQL transactions
        self._API_KEY = API_KEY  # Key for use with yahoo finance API
        self._alphaVantageBaseURL = "https://yfapi.net/v6/finance/quote" # base URL for yahoo
                
        # Dictionaries that converts a user-input value to the table/columns that 
        # should be accessed within the SQL database.  Helps to keep the 
        # database from being corrupted by user inputs.  First dict cooresponds
        # to the tables that align with the function calls.  The second is 
        # used to modify the values from the first dict to match the columns in 
        # the table 'ticker_symbol_list' (that holds a status board of sorts)
        self._tableConversion = commonUtilities.conversionTables.tickerConversionTable
        self._tickerListConversion = {"MOSTRECENT"  :  "date_",
                                      "ERRORCODE"   :  "error_",
                                      "NUMRECORDS"  :  "records_"}
        
        self.validate = commonUtilities.validationFunctions()
    
    
    
    def getDailyData(self, tickers = [], updateAll = False):
        if tickers == []:
            print("Loading Ticker List...")
            queryString  = "SELECT ticker_symbol FROM ticker_symbol_list \n"
            query = self._cur.execute(queryString)
            cols = [column[0] for column in query.description]
            results = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
            tickers = list(results["ticker_symbol"])
            print("Ticker List loaded...")
        
        if not updateAll:
            print("Removing currently added tickers...")
            queryString  = "SELECT ticker_symbol FROM yahoo_daily \n"
            query = self._cur.execute(queryString)
            cols = [column[0] for column in query.description]
            results = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
            existingTickers = list(results["ticker_symbol"])
            tickers = list(set(tickers) - set(existingTickers))
            tickers.sort()
            print("Ticker list reconciled.")
        
                    
        print("Removing unavailable tickers...")
        queryString  = "SELECT ticker_symbol FROM yahoo_missing \n"
        query = self._cur.execute(queryString)
        cols = [column[0] for column in query.description]
        results = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
        existingTickers = list(results["ticker_symbol"])
        tickers = list(set(tickers) - set(existingTickers))
        tickers.sort()
        print("Removed Missing Tickers.")
        
        
        count = 0
        totalCount = len(tickers)
        for ticker in tickers: 
            count += 1
            
            print("\rRequesting ticker:   " + str(ticker).rjust(6) + "   (" + str(count).rjust(6) + " of " + str(totalCount).ljust(6) + ")               ", end = "")
            url  = "https://query1.finance.yahoo.com/v7/finance/download/" + str(ticker)
            url += "?period1=315619200&period2=" + str(int(datetime.datetime.now().timestamp()))
            url += "&interval=1d&events=history&includeAdjustedClose=true"
            try:
                data = pd.read_csv(url)
            except Exception as e:
                print("\rError on ticker:     " + str(ticker).rjust(6) + ":     " + str(e) + ".                  ")
                if "Unauthorized" in str(e):
                    raise callLimitExceeded("API limit exceeded after " + str(count).rjust(6) + "   itterations on ticker:   " + str(ticker) + ". ")
                if "Not Found" in str(e):
                    self.insertMissing(ticker = ticker, error = "HTTPError: Not Found")
                continue
            
            if data.empty:
                print("\n   Ticker " + str(ticker).rjust(6) + "  empty.                                ")
                continue
            
            data.rename({"Open": "open",
                         "High": "high",
                         "Low": "low",
                         "Close": "close",
                         "Adj Close": "adj_close",
                         "Volume": "volume",
                         "Date": "recordDate"}, axis=1, inplace=True)
            
            data['ticker_symbol'] = str(ticker)
            
            # save the data to the SQLite database
            self.saveYahooToSQL(stockDF = data, ticker_symbol = ticker)
            
        print("\nData Collection Complete.")
        return False
        
        
        
        
    def saveYahooToSQL(self, stockDF, ticker_symbol = ""):
        # Saves a dataframe to the SQLite database.  Functions as a wrapper for
        # this class and its dataframes.  The dataframes and SQLite schema were
        # setup to match and work well together.
        
        # Create an entry for in the respective table for the API call that was made 
        insert_string  = "INSERT OR IGNORE INTO yahoo_daily "
        insert_string += " (ticker_symbol, recordDate) "
        insert_string += "VALUES(?, ?) \n"
        insert_list = stockDF[["ticker_symbol", "recordDate"]].values.tolist()
                
        self._cur.executemany(insert_string, insert_list)
        self.DB.stockDB.commit()
        
        
        # Update the record created above with the other data passed to the function.
        update_string = "UPDATE yahoo_daily SET \n"
        argList = pd.DataFrame()
        
        for key in stockDF.keys():
            if key not in ["ticker_symbol", "recordDate"]:
                update_string += "  " + key + " = ?,\n"
                argList[key] = stockDF[key]
        
        # Add 'where' clause to the SQL transaction to ensure that the correct record is altered.
        update_string = update_string[:-2] + "\nWHERE "
        update_string += "ticker_symbol = ? \n"
        update_string += "AND recordDate = ?; \n"
        argList["ticker_symbol"] = stockDF["ticker_symbol"]
        argList["recordDate"] = stockDF["recordDate"]
        
        argList = argList.values.tolist()
        
        self._cur.executemany(update_string, argList)
        self.DB.stockDB.commit()
    
    
    
    def insertMissing(self, ticker, error = ""):
        callTime = str(datetime.datetime.now().timestamp())
        callDate = str(datetime.date.today())
        
        insert_string  = "INSERT OR IGNORE INTO yahoo_missing "
        insert_string += " (ticker_symbol, error, call_time, call_date) "
        insert_string += "VALUES(?, ?, ?, ?) \n"
        insert_list = [ticker, error, callTime, callDate]
        
        self._cur.execute(insert_string, insert_list)
        self.DB.stockDB.commit()

    def sleep(self, interval = 0):
        print()
        while interval >= 0:
            print("\r sleeping for  " + str(interval).rjust(6) + "  seconds.              ", end = "")
            time.sleep(1)
            interval -= 1
        print()




class database:
    def __init__(self, dataBaseSaveFile = "SQLiteDB/stockData.db", dataBaseThreadCheck = True):
        if(os.path.exists(dataBaseSaveFile)):
            self.stockDB = sqlite3.connect(dataBaseSaveFile)
        else:
            try:
                self.stockDB = self.createStockDatabase(dataBaseSaveFile, check_same_thread = dataBaseThreadCheck)
                self.DBcursor = self.stockDB.cursor()
            except:
                print("Failed to connect to stock database.")
                sys.exit()
                
    
    def createStockDatabase(self, fileName):
        currentDir = os.getcwd()
        try:
            dirs = fileName.split("/")
            while('.' in dirs):
                dirs.remove('.')
            
            for i in range(len(dirs)-1):
                os.mkdir(dirs[i])
                os.chdir(dirs[i])
                
            os.chdir(currentDir)
            self.createSchema(fileName)
            
            return sqlite3.connect(fileName)
        
        except:
            print("Could not connect to database.")
            sys.exit()
    
    
    def createSchema(self, fileName):
        conn = sqlite3.connect(fileName)
        cur = conn.cursor()
        print("Opened database successfully.")
        
        
        cur.execute('''CREATE TABLE fundamental_overview (
                        id                         INTEGER     PRIMARY KEY     AUTOINCREMENT,
                        ticker_symbol              TEXT        NOT NULL,
                        recordDate                 TEXT        NOT NULL,
                        cik                        INTEGER     NOT NULL,
                        name                       TEXT        NOT NULL,
                        assetType                  TEXT,
                        description                TEXT,
                        exchange                   TEXT,
                        currency                   TEXT,
                        country                    TEXT,
                        sector                     TEXT,
                        industry                   TEXT,
                        address                    TEXT,
                        fiscalYearEnd              TEXT,
                        latestQuarter              TEXT,
                        marketCapitalization       INTEGER,
                        ebitda                     INTEGER,
                        peRatio                    REAL,
                        pegRatio                   REAL,
                        bookValue                  REAL,
                        dividendPerShare           REAL,
                        dividendYield              REAL,
                        eps                        REAL,
                        revenuePerShareTTM         REAL,
                        profitMargin               REAL,
                        operatingMarginTTM         REAL,
                        returnOnAssetsTTM          REAL,
                        returnOnEquityTTM          REAL,
                        revenueTTM                 INTEGER,
                        grossProfitTTM             INTEGER,
                        dilutedEPSTTM              REAL,
                        quarterlyEarningsGrowthYOY REAL,
                        quarterlyRevenueGrowthYOY  REAL,
                        analystTargetPrice         REAL,
                        trailingPE                 REAL,
                        forwardPE                  REAL,
                        priceToSalesRatioTTM       REAL,
                        priceToBookRatio           REAL,
                        evToRevenue                REAL,
                        evToEBITDA                 REAL,
                        beta                       REAL,
                        high_52week                REAL,
                        low_52week                 REAL,
                        moving_average_50d         REAL,
                        moving_average_200d        REAL,
                        sharesOutstanding          INTEGER,
                        sharesFloat                INTEGER,
                        sharesShort                INTEGER,
                        sharesShortPriorMonth      INTEGER,
                        shortRatio                 REAL,
                        shortPercentOutstanding    REAL,
                        shortPercentFloat          REAL,
                        percentInsiders            REAL,
                        percentInstitutions        REAL,
                        forwardAnnualDividendRate  REAL,
                        forwardAnnualDividendYield REAL,
                        payoutRatio                REAL,
                        lastSplitFactor            REAL,
                        lastSplitDate              TEXT,
                        dividendDate               TEXT,
                        exDividendDate             TEXT,
                        CONSTRAINT ticker_date_constraint UNIQUE (
                            ticker_symbol ASC,
                            recordDate
                        )
                        ON CONFLICT IGNORE);  ''')
        
        
                
        cur.execute('''CREATE TABLE daily_adjusted (
                        id                INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker_symbol     TEXT,
                        recordDate        TEXT,
                        open              REAL,
                        high              REAL,
                        low               REAL,
                        close             REAL,
                        adj_close         REAL,
                        volume            REAL,
                        vol_avg_20        REAL,
                        vol_avg_50        REAL,
                        dividend          REAL,
                        split             REAL,
                        adjustment_ratio  REAL,
                        percent_cng_day   REAL,
                        percent_cng_tot   REAL,
                        mvng_avg_20       REAL,
                        tp20              REAL,
                        bollinger_20      REAL,
                        mvng_avg_20_trig  INTEGER,
                        bollinger_20_trig REAL,
                        mvng_avg_50       REAL,
                        tp50              REAL,
                        bollinger_50      REAL,
                        mvng_avg_50_trig  INTEGER,
                        bollinger_50_trig REAL,
                        macd_12_26        REAL,
                        macd_12_26_trig   INTEGER,
                        macd_19_39        REAL,
                        macd_19_39_trig   INTEGER,
                        on_bal_vol        REAL,
                        on_bal_vol_trig   INTEGER,
                        rsi               REAL,
                        rsi_trig          INTEGER,
                        ideal_high        REAL,
                        ideal_low         REAL,
                        ideal_return      REAL,
                        ideal_return_trig INTEGER,
                        CONSTRAINT ticker_date_constraint UNIQUE (
                            ticker_symbol ASC,
                            recordDate
                        )
                        ON CONFLICT IGNORE); ''')

        
        
        cur.execute('''CREATE TABLE income_statement (
                        id                                INTEGER     PRIMARY KEY     AUTOINCREMENT,
                        ticker_symbol                     TEXT        NOT NULL,
                        recordDate                        TEXT        NOT NULL,
                        annualReport                      INTEGER     NOT NULL,
                        reportedCurrency                  TEXT,
                        grossProfit                       REAL,
                        totalRevenue                      INTEGER,
                        costOfRevenue                     INTEGER,
                        costofGoodsAndServicesSold        INTEGER,
                        operatingIncome                   INTEGER,
                        sellingGeneralAndAdministrative   INTEGER,
                        researchAndDevelopment            INTEGER,
                        operatingExpenses                 INTEGER,
                        investmentIncomeNet               INTEGER,
                        netInterestIncome                 INTEGER,
                        interestIncome                    INTEGER,
                        interestExpense                   INTEGER,
                        nonInterestIncome                 INTEGER,
                        otherNonOperatingIncome           INTEGER,
                        depreciation                      INTEGER,
                        depreciationAndAmortization       INTEGER,
                        incomeBeforeTax                   INTEGER,
                        incomeTaxExpense                  INTEGER,
                        interestAndDebtExpense            INTEGER,
                        netIncomeFromContinuingOperations INTEGER,
                        comprehensiveIncomeNetOfTax       INTEGER,
                        ebit                              INTEGER,
                        ebitda                            INTEGER,
                        netIncome                         INTEGER,
                        CONSTRAINT ticker_date_report_constraint UNIQUE (
                            ticker_symbol ASC,
                            recordDate,
                            annualReport
                        )
                        ON CONFLICT IGNORE);  ''')
                 
        
        
        cur.execute('''CREATE TABLE balance_sheet (
                        id                                     INTEGER     PRIMARY KEY     AUTOINCREMENT,
                        ticker_symbol                          TEXT        NOT NULL,
                        recordDate                             TEXT        NOT NULL,
                        annualReport                           INTEGER     NOT NULL,
                        reportedCurrency                       TEXT,
                        totalAssets                            INTEGER,
                        totalCurrentAssets                     INTEGER,
                        cashAndCashEquivalentsAtCarryingValue  INTEGER,
                        cashAndShortTermInvestments            INTEGER,
                        inventory                              INTEGER,
                        currentNetReceivables                  INTEGER,
                        totalNonCurrentAssets                  INTEGER,
                        propertyPlantEquipment                 INTEGER,
                        accumulatedDepreciationAmortizationPPE INTEGER,
                        intangibleAssets                       INTEGER,
                        intangibleAssetsExcludingGoodwill      INTEGER,
                        goodwill                               INTEGER,
                        investments                            INTEGER,
                        longTermInvestments                    INTEGER,
                        shortTermInvestments                   INTEGER,
                        otherCurrentAssets                     INTEGER,
                        otherNonCurrrentAssets                 INTEGER,
                        totalLiabilities                       INTEGER,
                        totalCurrentLiabilities                INTEGER,
                        currentAccountsPayable                 INTEGER,
                        deferredRevenue                        INTEGER,
                        currentDebt                            INTEGER,
                        shortTermDebt                          INTEGER,
                        totalNonCurrentLiabilities             INTEGER,
                        capitalLeaseObligations                INTEGER,
                        longTermDebt                           INTEGER,
                        currentLongTermDebt                    INTEGER,
                        longTermDebtNoncurrent                 INTEGER,
                        shortLongTermDebtTotal                 INTEGER,
                        otherCurrentLiabilities                INTEGER,
                        otherNonCurrentLiabilities             INTEGER,
                        totalShareholderEquity                 INTEGER,
                        treasuryStock                          INTEGER,
                        retainedEarnings                       INTEGER,
                        commonStock                            INTEGER,
                        commonStockSharesOutstanding           INTEGER,
                        CONSTRAINT ticker_date_report_constraint UNIQUE (
                            ticker_symbol ASC,
                            recordDate,
                            annualReport
                        )
                        ON CONFLICT IGNORE);  ''')



        cur.execute('''CREATE TABLE earnings (
                        id                 INTEGER     PRIMARY KEY     AUTOINCREMENT,
                        ticker_symbol      TEXT        NOT NULL,
                        recordDate         TEXT        NOT NULL,
                        reportedDate       TEXT,
                        reportedEPS        INTEGER,
                        estimatedEPS       INTEGER,
                        surprise           INTEGER,
                        surprisePercentage INTEGER,
                        CONSTRAINT ticker_date_constraint UNIQUE (
                            ticker_symbol ASC,
                            recordDate
                        )
                        ON CONFLICT IGNORE);   ''')



        cur.execute('''CREATE TABLE cash_flow (
                        id                                    INTEGER     PRIMARY KEY     AUTOINCREMENT,
                        ticker_symbol                         TEXT        NOT NULL,
                        recordDate                            TEXT        NOT NULL,
                        annualReport                          INTEGER     NOT NULL,
                        reportedCurrency                      TEXT,
                        operatingCashflow                     INTEGER,
                        paymentsForOperatingActivities        INTEGER,
                        proceedsFromOperatingActivities       INTEGER,
                        changeInOperatingLiabilities          INTEGER,
                        changeInOperatingAssets               INTEGER,
                        depreciationDepletionAndAmortization  INTEGER,
                        capitalExpenditures                   INTEGER,
                        changeInReceivables                   INTEGER,
                        changeInInventory                     INTEGER,
                        profitLoss                            INTEGER,
                        cashflowFromInvestment                INTEGER,
                        cashflowFromFinancing                 INTEGER,
                        proceedsFromRepaymentsOfShortTermDebt INTEGER,
                        paymentsForRepurchaseOfCommonStock    INTEGER,
                        paymentsForRepurchaseOfEquity         INTEGER,
                        paymentsForRepurchaseOfPreferredStock INTEGER,
                        dividendPayout                        INTEGER,
                        dividendPayoutCommonStock             INTEGER,
                        dividendPayoutPreferredStock          INTEGER,
                        proceedsFromIssuanceOfCommonStock     INTEGER,
                        proceedsFromIssuanceOfLongTermDebt    INTEGER,
                        proceedsFromIssuanceOfPreferredStock  INTEGER,
                        proceedsFromRepurchaseOfEquity        INTEGER,
                        proceedsFromSaleOfTreasuryStock       INTEGER,
                        changeInCashAndCashEquivalents        INTEGER,
                        changeInExchangeRate                  INTEGER,
                        netIncome                             INTEGER,
                        CONSTRAINT ticker_date_report_constraint UNIQUE (
                            ticker_symbol ASC,
                            recordDate,
                            annualReport
                        )
                        ON CONFLICT IGNORE  );''')
        
        
        
        cur.execute('''CREATE TABLE api_transactions (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker_symbol TEXT    NOT NULL,
                        call_time     REAL    NOT NULL,
                        call_date     TEXT,
                        type          TEXT,
                        source        TEXT,
                        message       TEXT,
                        status        TEXT,
                        CONSTRAINT ticker_time_constraint UNIQUE (
                            ticker_symbol,
                            call_time
                        )
                        ON CONFLICT FAIL );''')
        
        
        
        cur.execute('''CREATE TABLE ticker_symbol_list (
                        id                           INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker_symbol                TEXT    NOT NULL,
                        recordDate                   TEXT    NOT NULL,
                        name                         TEXT,
                        exchange                     TEXT,
                        error_fundamental_overview   INTEGER DEFAULT -1     NOT NULL,
                        error_income_statement       INTEGER DEFAULT -1     NOT NULL,
                        error_earnings               INTEGER DEFAULT -1     NOT NULL,
                        error_alpha_daily            INTEGER DEFAULT -1     NOT NULL,
                        error_balance_sheet          INTEGER DEFAULT -1     NOT NULL,
                        error_cash_flow              INTEGER DEFAULT -1     NOT NULL,
                        date_fundamental_overview    TEXT,
                        date_income_statement        TEXT,
                        date_earnings                TEXT,
                        date_alpha_daily             TEXT,
                        date_balance_sheet           TEXT,
                        date_cash_flow               TEXT,
                        records_fundamental_overview INTEGER,
                        records_income_statement     INTEGER,
                        records_earnings             INTEGER,
                        records_alpha_daily          INTEGER,
                        records_balance_sheet        INTEGER,
                        records_cash_flow            INTEGER,
                        CONSTRAINT ticker_constraint UNIQUE (
                            ticker_symbol ASC
                        )
                        ON CONFLICT IGNORE);  ''')
        
        
        
        cur.execute('''CREATE TABLE summary_data (
                        id                  INTEGER      PRIMARY KEY      AUTOINCREMENT      UNIQUE,
                        ticker_symbol       TEXT         NOT NULL,
                        date_calculated     TEXT         NOT NULL,
                        daily_length        INTEGER,
                        market_cap          REAL,
                        sector              TEXT,
                        exchange            TEXT,
                        country             TEXT,
                        industry            TEXT,
                        pe_ratio            REAL,
                        profit_margin       REAL,
                        share_holder_equity REAL,
                        earnings_per_share  REAL,
                        avg_return          REAL,
                        std_return          REAL,                    
                        comp_return         REAL,
                        comp_stddev         REAL,                    
                        max_daily_change    REAL,
                        min_daily_change    REAL,
                        CONSTRAINT ticker_constraint UNIQUE (ticker_symbol ASC)
                        ON CONFLICT IGNORE );  ''')

        
        print("Schema created successfully.")
        
        conn.close()



if __name__ == "__main__":
    t_start = time.time()
    
    info = getAlphaVantageData()
    
    # info.confirmDatesAndCounts()
    
    # TIME_SERIES_DAILY_ADJUSTED is a paid option for function _getTimeSeriesDaily()
    # Now using TIME_SERIES_DAILY instead.  May consider using weekly adjusted 
    # as a way to get the adjustment value.
    
    # complete = info.autoUpdate(missing = False)
    
    # while(complete != 0):
    #     try:
    #         complete = info.autoUpdate()
    #     except callLimitExceeded:
    #         pass  # Alpha vantage would track IP addresses, so resetting a VPN here
    #               # would, at one point, allow a user to get more data than their 
    #               # API would otherwise allow.
    
    # t_1 = time.time()


    # info = getYahooData()
    
    # notDone = True
    
    # while notDone:
    #     try:
    #         notDone = info.getDailyData()
    #     except callLimitExceeded:
    #         print(callLimitExceeded)
    #         info.sleep(300)









