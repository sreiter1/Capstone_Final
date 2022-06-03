# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:00:28 2021

@author: sreit
"""

import pandas as pd
import numpy as np
import sqlite3
import commonUtilities
import matplotlib.pyplot as plt

#import statsmodels.api as sm


import warnings
warnings.filterwarnings("ignore")


class missingTicker(Exception):
    pass



class analysis:
    def __init__(self, dataBaseSaveFile = "./stockData.db"):
        self.mainDBName = dataBaseSaveFile
        self.DB = sqlite3.connect(dataBaseSaveFile)
        self._cur = self.DB.cursor()
        self._tickerList = []   # Empty list that gets filled with a list of tickers to be considered
        self.tradingDateSet = []  # List of dates in YYYY-MM-DD format that are trading dates in the database
        self.dailyTableNames = ["alpha", "yahoo"]
        
        self.validate = commonUtilities.validationFunctions()
        
        # Converts a user string to the names of tables, and for 'ticker_symbol_list'
        self._tickerConversionTable = commonUtilities.conversionTables.tickerConversionTable
        
        # Converts user unput to the columns in the table.  Provides a filter to 
        # prevent database corruption.
        self._dailyConversionTable = commonUtilities.conversionTables.dailyConversionTable
        
        self.indicatorList = {"ADJRATIO"     : "adjustment_ratio",
                              "MA20"         : "mvng_avg_20", 
                              "MA50"         : "mvng_avg_50", 
                              "BOLLINGER20"  : "bollinger_20",
                              "TP20"         : "tp20",
                              "BOLLINGER50"  : "bollinger_50",
                              "TP50"         : "tp50",
                              "MACD12"       : "macd_12_26", 
                              "MACD19"       : "macd_19_39",
                              "VOL20"        : "vol_avg_20",
                              "VOL50"        : "vol_avg_50",
                              "OBV"          : "on_bal_vol", 
                              "DAYCHANGE"    : "percent_cng_day", 
                              "TOTALCHANGE"  : "percent_cng_tot", 
                              "RSI"          : "rsi"}
        
    
    
    def dailyReturns(self,
                     tickerList = [],
                     plot = False):
        
        self.validate.validateListString(tickerList)
        
        tickerGroups = int(len(tickerList) / 100) + 1
        results = pd.DataFrame()
        
        for i in range(tickerGroups):
            minIndex = i * 100
            maxIndex = minIndex + 100 if (minIndex + 100 < len(tickerList)) else len(tickerList) - 1
            
            tickerListSubset = tickerList[minIndex:maxIndex]
            
            # start a SQL query string and an list of arguments.  Leave a space for
            # the specific columns that should be requested ("[REQUEST_WHAT]")
            queryString = "SELECT * " +\
                          "FROM summary_data \n" +\
                          "WHERE 1=1  \n AND "
            
            argList = []
            
            # append to the SQL query string and argument list each ticker from the 
            # function inputs
            for ticker in tickerListSubset:
                ticker = "".join(e.upper() for e in ticker if e.isalpha())
                argList.append(ticker)
                queryString += "ticker_symbol = ? \n  OR "
            
            # sort the tickers aphabetically, convert the argument list to a non-changeable tuple,
            # and complete the SQL query string
            argList.sort()
            argList = tuple(argList)
            queryString = queryString[:-7] + ";\n"
        
        
            # execute the SQL query and convert the response to a pandas dataframe
            query = self._cur.execute(queryString, argList)
            cols = [column[0] for column in query.description]
            results = results.append(pd.DataFrame.from_records(data = query.fetchall(), columns = cols))
            
        
        if plot == True and len(tickerList) < 10:
            for ticker in tickerListSubset:
                queryString  = "SELECT * "
                queryString += "FROM daily_adjusted \n"
                queryString += "WHERE ticker_symbol = '" + ticker + "';\n"
                
                query = self._cur.execute(queryString)
                cols = [column[0] for column in query.description]
                plotResults = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
                
                # convert the recordDate from a string to a datetime object, and set as the index
                plotResults["recordDate"] = pd.to_datetime(plotResults["recordDate"])
                plotResults = plotResults.set_index("recordDate")
                plotResults.sort_index()
                
                plt.figure()
                plotResults["percent_cng_day"].hist(bins = 1000)
                plt.figure()
                plotResults["adj_close"].plot()
        
        
        results = results.reset_index()
        return results
    
    
    
    def compareDataSources(self, ticker):
        self.validate.validateString(ticker)
        
        df = []
        for i in range(len(self.dailyTableNames)):
            table = self.dailyTableNames[i]
            queryString  = "SELECT recordDate, adj_close, volume "
            queryString += "FROM " + table + "_daily \n"
            queryString += "WHERE ticker_symbol = ?\n"
            
            query = self._cur.execute(queryString, [ticker])
            cols = [column[0] for column in query.description]
            df.append(pd.DataFrame.from_records(data = query.fetchall(), columns = cols))
            if df[i]["recordDate"].empty:
                raise missingTicker("Ticker   " + str(ticker).rjust(6) + "   Missing from table   " + str(table) + "_daily.")
        
        
        columnNames = ["missing_dates", "entries", "startDate", 
                       "abs_delta_adj_close", "abs_delta_volume", 
                       "percent_delta_close", "percent_delta_volume"]
        stats = pd.DataFrame(columns = columnNames, index = self.dailyTableNames)
        
        for i in range(len(self.dailyTableNames)):
            table = self.dailyTableNames[i]
            totalDates = [n for n in self.tradingDateSet if n >= df[i]["recordDate"].min()]
            totalDates = set(totalDates)
            tickerDates = set(df[i]["recordDate"])
            missingDates = list(totalDates - tickerDates)
            missingDates.sort()
            stats.at[table, "missing_dates"] = missingDates
            
            stats.at[table, "entries"] = df[i]["recordDate"].count()
            stats.at[table, "startDate"] = df[i]["recordDate"].min()
            
            if i == 0:
                stats.at[table, "abs_delta_adj_close"] = 0
                stats.at[table, "abs_delta_volume"] = 0
                stats.at[table, "percent_delta_close"] = 0
                stats.at[table, "percent_delta_volume"] = 0
            else:
                stats.at[table, "abs_delta_adj_close"] = [abs(a - b) for a, b in zip(df[0]["adj_close"], df[i]["adj_close"])]
                stats.at[table, "abs_delta_volume"] = [abs(a - b) for a, b in zip(df[0]["volume"], df[i]["volume"])]
                stats.at[table, "percent_delta_close"] = [100*abs(a - b) / a for a, b in zip(df[0]["adj_close"], df[i]["adj_close"])]
                stats.at[table, "percent_delta_volume"] = [100*abs(a - b) / a for a, b in zip(df[0]["volume"], df[i]["volume"])]
        
        return stats
        
        
        
    def fillTradingDates(self):
        
        for table in self.dailyTableNames:
            print("\rCollecting dates from table:  " + table + "                      ", end = "")
            queryString  = "SELECT DISTINCT recordDate "
            queryString += "FROM " + table + "_daily \n"
            
            query = self._cur.execute(queryString)
            cols = [column[0] for column in query.description]
            tickerList = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
            tickerList = list(tickerList["recordDate"])
            self.tradingDateSet += tickerList
        
        print("\rSorting Data...                                                ", end = "")
        self.tradingDateSet = set(self.tradingDateSet)
        self.tradingDateSet = list(self.tradingDateSet)
        self.tradingDateSet.sort()
        saveArray = [[x] for x in self.tradingDateSet]
        
        print("\rSaving Data...                                                ", end = "")
        
        query  = "INSERT OR IGNORE INTO trading_dates (trading_date)\n"
        query += "VALUES(?)"
        self._cur.executemany(query, saveArray)
        self.DB.commit()
        
        return
    
    
    
    def filterStocksFromDataBase(self, 
                                 dailyLength = 0,            # time = 1250 is 5 years of trading days.  Default is keep all.
                                 marketCap = 0,              # commonStockSharesOutstanding from balance * price from daily
                                 sector = None,              # fundamental overview
                                 exchange = None,            # fundamental overview
                                 country = None,             # fundamental overview
                                 industry = None,            # fundamental overview
                                 peRatio = 0.0,              # EPS under earnings, price under daily
                                 profitMargin = 0.0,         # profit/loss in cash flow; net income / total revenue --> income statement
                                 shareHolderEquity = 0,      # balance sheet
                                 EPS = 0.0,                  # from earnings
                                 maxDailyChange = None,      # maximum daily % change
                                 minDailyChange = None,      # minimum daily % change
                                 minDailyVolume = None       # minimum daily volume; 
                                 ):     
                       
        # Creates a list of stocks that meet the requirements passed to the function.
        
        # check inputs
        self.validate.validateInteger(dailyLength)
        self.validate.validateInteger(marketCap)
        self.validate.validateInteger(shareHolderEquity)
        
        self.validate.validateNum(peRatio)
        self.validate.validateNum(profitMargin)
        self.validate.validateNum(EPS)
        if maxDailyChange != None:
            self.validate.validateNum(maxDailyChange)
        if minDailyChange != None:
            self.validate.validateNum(minDailyChange)
        if minDailyVolume != None:
            self.validate.validateNum(minDailyVolume)
        
        
        if sector is None: pass
        else: self.validate.validateListString(sector)
        
        if exchange is None: pass
        else: self.validate.validateListString(exchange)
        
        if country is None: pass
        else: self.validate.validateListString(country)
        
        if industry is None: pass
        else: self.validate.validateListString(industry)
        
        
        # start SQL query string.  "WHERE 1=1 AND ..." allows for the addition 
        # of other requirements within the request string later
        query  = "SELECT [COLUMNS]"
        query += "FROM summary_data \n"
        query += "WHERE 1=1 [CONDITIONS]"
                
        # create strings that will be inserted into the query string later, as
        # well as an empty list for the arguments that will also be passed
        columnString = "ticker_symbol, \n  "
        conditionString = "\n  AND "
        argList = []
        
        # add requirements for the shortest price history length that will be allowed
        if dailyLength != 0:
            argList.append(str(dailyLength))
            columnString += "daily_length, \n  "
            conditionString += "daily_length >= ? \n  AND "
            
        # add requirements for the minimum market capitalization that will be allowed
        if marketCap != 0:
            argList.append(str(marketCap))
            columnString += "market_capitalization, \n  "
            conditionString += "market_capitalization >= ? \n  AND "
            
        # add requirements for the minimum PE ratio that will be allowed
        if peRatio != 0:
            argList.append(str(peRatio))
            columnString += "pe_ratio, \n  "
            conditionString += "pe_ratio >= ? \n  AND "
            
        # add requirements for the minimum profit margin that will be allowed
        if profitMargin != 0:
            argList.append(str(profitMargin))
            columnString += "profit_margin, \n  "
            conditionString += "profit_margin >= ? \n  AND "
            
        # add requirements for the minimum shareholder equity that will be allowed
        if shareHolderEquity != 0:
            argList.append(str(shareHolderEquity))
            columnString += "share_holder_equity, \n  "
            conditionString += "share_holder_equity >= ? \n  AND "
            
        # add requirements for the minimum earnings per share that will be allowed
        if EPS != 0:
            argList.append(str(EPS))
            columnString += "earnings_per_share, \n  "
            conditionString += "earnings_per_share >= ? \n  AND "
        
        # add requirements for the maximum daily change that will be allowed
        if maxDailyChange != None:
            maxDailyChange = abs(maxDailyChange)
            argList.append(str(maxDailyChange))
            columnString += "max_daily_change, \n  "
            conditionString += "max_daily_change <= ? \n  AND "
        
        # add requirements for the maximum daily change that will be allowed
        if minDailyChange != None:
            minDailyChange = -abs(minDailyChange)
            argList.append(str(minDailyChange))
            columnString += "min_daily_change, \n  "
            conditionString += "min_daily_change >= ? \n  AND "
        
        # add requirements for the maximum daily change that will be allowed
        if minDailyVolume != None:
            minDailyVolume = abs(minDailyVolume)
            argList.append(str(minDailyVolume))
            columnString += "min_daily_volume, \n  "
            conditionString += "min_daily_volume >= ? \n  AND "
        
        
        # add requirements for the sectors that will be allowed
        if sector is not None:
            conditionString += " ("
            for sect in sector:
                argList.append(str(sect))
                columnString += "sector, \n  "
                conditionString += "sector = ? \n OR "
            
            conditionString = conditionString[:-5] + " ) \n  AND "
            
        
        # add requirements for the exchanges that will be allowed
        if exchange is not None:
            conditionString += " ("
            for ex in exchange:
                argList.append(str(ex))
                columnString += "exchange, \n  "
                conditionString += "exchange = ? \n OR "
            
            conditionString = conditionString[:-5] + " ) \n  AND "
            
        
        # add requirements for the exchanges that will be allowed
        if country is not None:
            conditionString += " ("
            for co in country:
                argList.append(str(co))
                columnString += "country, \n  "
                conditionString += "country = ? \n OR "
            
            conditionString = conditionString[:-5] + " ) \n  AND "
            
        
        # add requirements for the exchanges that will be allowed
        if industry is not None:
            conditionString += " ("
            for ind in industry:
                argList.append(str(ind))
                columnString += "industry, \n  "
                conditionString += "industry = ? \n OR "
            
            conditionString = conditionString[:-5] + " ) \n  AND "
            
        
        # remove extra characters at the end of the string to format it correctly
        # for the SQL query
        columnString = columnString[:-5] + " \n"
        conditionString = conditionString[:-7] + "; \n"
        
        # replace the placeholders in the original query string with specific 
        # values
        query = query.replace("[COLUMNS]", columnString)
        query = query.replace("[CONDITIONS]", conditionString)
        
        # execute the SQL query and format the response as a pandas dataframe
        result = self._cur.execute(query, argList)
        cols = [column[0] for column in result.description]
        DF_tickerList = pd.DataFrame.from_records(data = result.fetchall(), columns = cols)
        
        # update the internal list of tickers to match those that are in the response
        self._tickerList = list(DF_tickerList["ticker_symbol"])
        
        if self._tickerList == []:
            warnings.warn("tickerList is an empty list.  Is the table 'summary_data' empty?  Run filter fuction with option updateBeforeFilter = True")
        
        # return the dataframe
        return DF_tickerList
    
    
    
    def listUnique(self, extended = False):
        # returns a dictionary of the unique values available in the summary_data table
        
        # start SQL query string.  "WHERE 1=1 AND ..." allows for the addition 
        # of other requirements within the request string later
        query  = "SELECT [COLUMNS]"
        query += "FROM summary_data \n"
                
        # create strings that will be inserted into the query string later, as
        # well as an empty list for the arguments that will also be passed
        columnString = "ticker_symbol, \n  "
        
        # populate the columns to be gathered
        if extended == True:
            columnString += "date_calculated, \n  "
            columnString += "daily_length, \n  "
            columnString += "earnings_per_share, \n  "
            columnString += "profit_margin, \n  "
            columnString += "share_holder_equity, \n  "
            columnString += "common_shares_outstanding, \n  "
            columnString += "current_price, \n  "
            columnString += "market_capitalization, \n  "
            columnString += "pe_ratio, \n  "
            columnString += "avg_return, \n  "
            columnString += "std_return, \n  "
            columnString += "comp_return, \n  "
            columnString += "comp_stddev, \n  "
            columnString += "max_daily_change, \n  "
            columnString += "min_daily_change, \n  "
            columnString += "min_daily_volume, \n  "
        
        
        columnString += "country, \n  "
        columnString += "exchange, \n  "
        columnString += "sector, \n  "
        columnString += "industry \n  "
        
        # execute the SQL query and format the response as a pandas dataframe
        query = query.replace("[COLUMNS]", columnString)
        result = self._cur.execute(query)
        cols = [column[0] for column in result.description]
        df = pd.DataFrame.from_records(data = result.fetchall(), columns = cols)
        
        self.uniqueValues = {}
        self.uniqueValueCounts = {}
        for colName in df:
            self.uniqueValues[colName] = df[colName].unique()
            countList = []
            for countTuple in df[colName].value_counts().iteritems():
                countList.append(countTuple)
            self.uniqueValueCounts[colName] = countList
            
        return df
    
    
    
    def coor_MACD(self, tickerList = [], tradeDelay = 0):
        indics = ["MACD12", "MACD19"]
        loadedData, trigList = self.loadIndicatorFromDB(tickerList = tickerList, indicators = indics)
        
        loadedData["tradePrice"] = None
        for ind in indics:
            loadedData[self.indicatorList[ind] + "_trig"] = None
        
        for ind in indics:
            indColName = self.indicatorList[ind]
            results = pd.DataFrame(columns = loadedData.columns)
            
            x_p = []
            x_n = []
            y_p = []
            y_n = []
            
            for tick in tickerList:
                print("\rProcessing ticker:  " + str(tick).rjust(6) + "  for  " + ind + ".                                       ", end = "")
                rslt_df = loadedData.loc[loadedData["ticker_symbol"] == tick]
                rslt_df[indColName + "_trig"] = np.append(np.diff(np.sign(rslt_df[indColName])), [0])
                rslt_df["tradePrice"] = rslt_df["adj_close"].shift(periods = tradeDelay)
                
                results = pd.concat([results, rslt_df])
            
            
            print("\rPlotting  for  " + ind + ".                                                                      ", end = "")
            triggerLoc = list(np.where(np.diff(np.sign(results[indColName])))[0])
            
            for i in range(len(triggerLoc)-1):
                if results[indColName + "_trig"][triggerLoc[i]] > 0:
                    y_p.append(list(results["tradePrice"][triggerLoc[i]+1:triggerLoc[i+1]]/results["tradePrice"][triggerLoc[i]]))
                    x_p.append(list(range(triggerLoc[i+1] - triggerLoc[i] - 1)))
                else:
                    y_n.append(list(results["tradePrice"][triggerLoc[i]+1:triggerLoc[i+1]]/results["tradePrice"][triggerLoc[i]]))
                    x_n.append(list(range(triggerLoc[i+1] - triggerLoc[i] - 1)))
            
            
            x_p = pd.Series([x for seg in x_p for x in seg])
            x_n = pd.Series([x for seg in x_n for x in seg])
            
            y_p = pd.Series([y/seg[0]-1 for seg in y_p for y in seg])
            y_n = pd.Series([y/seg[0]-1 for seg in y_n for y in seg])
            
            plt.figure()
            plt.scatter(x_p, y_p, marker = ".", s = 5, c = "#00cc00", label = "buy")
            plt.scatter(x_n, y_n, marker = ".", s = 5, c = "#ff0000", label = "sell")
            plt.title("Scatter Plot of time vs returns " + ind)
            plt.legend()
            
            
            avgNeg = y_n.mean()
            avgPos = y_p.mean()
            avgLen = len(results)/len(triggerLoc)
            
            print()
            print()
            print("average " + ind + " positive return:   " + str(avgPos))
            print("average " + ind + " negative return:   " + str(avgNeg))
            print("average " + ind + " timeframe:         " + str(avgLen))
            print()
            
        return results, triggerLoc
    
    
    
    def coor_MA(self, tickerList = [], tradeDelay = 0):
        indics = ["MA20", "MA50"]
        loadedData, trigList = self.loadIndicatorFromDB(tickerList = tickerList, indicators = indics)
        
        loadedData["tradePrice"] = None
        for ind in indics:
            loadedData[self.indicatorList[ind] + "_trig"] = None
        
        for ind in indics:
            indColName = self.indicatorList[ind]
            results = pd.DataFrame(columns = loadedData.columns)
            
            x_p = []
            x_n = []
            y_p = []
            y_n = []
            
            for tick in tickerList:
                print("\rProcessing ticker:  " + str(tick).rjust(6) + "  for " + ind + ".                                       ", end = "")
                rslt_df = loadedData.loc[loadedData["ticker_symbol"] == tick]
                rslt_df[indColName + "_trig"] = np.append(np.diff(np.sign(rslt_df["adj_close"] - rslt_df[indColName])), [0])
                rslt_df["tradePrice"] = rslt_df["adj_close"].shift(periods = tradeDelay)
                
                results = pd.concat([results, rslt_df])
            
            
            print("\rPlotting  for  " + ind + ".                                                                      ", end = "")
            triggerLoc = list(np.where(np.diff(np.sign(results[indColName + "_trig"])))[0])
            
            for i in range(len(triggerLoc)-1):
                if results[indColName + "_trig"][triggerLoc[i]] > 0:
                    y_p.append(list(results["tradePrice"][triggerLoc[i]+1:triggerLoc[i+1]]/results["tradePrice"][triggerLoc[i]]))
                    x_p.append(list(range(triggerLoc[i+1] - triggerLoc[i] - 1)))
                else:
                    y_n.append(list(results["tradePrice"][triggerLoc[i]+1:triggerLoc[i+1]]/results["tradePrice"][triggerLoc[i]]))
                    x_n.append(list(range(triggerLoc[i+1] - triggerLoc[i] - 1)))
            
            
            x_p = pd.Series([x for seg in x_p for x in seg])
            x_n = pd.Series([x for seg in x_n for x in seg])
            
            y_p = pd.Series([y/seg[0]-1 for seg in y_p for y in seg])
            y_n = pd.Series([y/seg[0]-1 for seg in y_n for y in seg])
            
            plt.figure()
            plt.scatter(x_p, y_p, marker = ".", s = 5, c = "#00cc00", label = "buy")
            plt.scatter(x_n, y_n, marker = ".", s = 5, c = "#ff0000", label = "sell")
            plt.title("Scatter Plot of time vs returns " + ind)
            plt.legend()
            
            
            avgNeg = y_n.mean()
            avgPos = y_p.mean()
            avgLen = len(results)/len(triggerLoc)
            
            print("average " + ind + " positive return:   " + str(avgPos))
            print("average " + ind + " negative return:   " + str(avgNeg))
            print("average " + ind + " timeframe:         " + str(avgLen))
            print()
            
        return results, triggerLoc
    
    
    
    def coor_BOLL(self, tickerList = [], tradeDelay = 0):
        # need the TP for Bollinger Band Calcs
        indics = {"BOLLINGER20": "TP20", 
                  "BOLLINGER50": "TP50"}
        loadedData, trigList = self.loadIndicatorFromDB(tickerList = tickerList, 
                                                        indicators = list(indics.keys()) + list(indics.values()))
        
        loadedData["tradePrice"] = None
        
        for ind in indics.keys():
            loadedData[self.indicatorList[ind] + "_trig"] = None
        
        for ind in indics.keys():
            indColName  = self.indicatorList[ind]
            helpColName = self.indicatorList[indics[ind]]
            results = pd.DataFrame(columns = loadedData.columns)
            
            x_p = []
            x_n = []
            y_p = []
            y_n = []
            
            for tick in tickerList:
                print("\rProcessing ticker:  " + str(tick).rjust(6) + "  for " + ind + ".                                                            ", end = "")
                rslt_df = loadedData.loc[loadedData["ticker_symbol"] == tick]
                
                upper = [tp + b for tp, b in zip(rslt_df[helpColName], rslt_df[indColName])]
                lower = [tp - b for tp, b in zip(rslt_df[helpColName], rslt_df[indColName])]
                trig  = [0] * len(upper)
                delta = list(np.sign([t2 - t1 for t2, t1 in zip(rslt_df[helpColName][1:], rslt_df[helpColName][:-1])]))
                delta.append(0)
                lstlen=len(delta)
                trend = np.sign([sum(delta[max(i-20, 0):i]) for i in range(lstlen)])
                
                trig = [-1 if c > u and tr < 0 else tg for c, u, tr, tg in zip(rslt_df["adj_close"], upper, trend, trig)]
                trig = [ 1 if c < l and tr > 0 else tg for c, l, tr, tg in zip(rslt_df["adj_close"], lower, trend, trig)]
                
                rslt_df[indColName + "_trig"] = trig
                rslt_df["tradePrice"] = rslt_df["adj_close"].shift(periods = tradeDelay)
                
                results = pd.concat([results, rslt_df])
            
            
            print("\rPlotting  for  " + ind + ".                                                                      ", end = "")
            triggerLoc = [i for i, e in enumerate(results[indColName + "_trig"]) if e != 0]
            # triggerLoc = []
            
            # return loc, results[indColName + "_trig"], upper
            
            # for i in range(1,len(loc)):
            #     if results[indColName + "_trig"][loc[i]] != results[indColName + "_trig"][loc[i-1]]:
            #         triggerLoc.append(loc[i])
                    
            
            for i in range(len(triggerLoc)-1):
                if results[indColName + "_trig"][triggerLoc[i]] > 0:
                    y_p.append(list(results["tradePrice"][triggerLoc[i]+1:triggerLoc[i+1]]/results["tradePrice"][triggerLoc[i]]))
                    x_p.append(list(range(triggerLoc[i+1] - triggerLoc[i] - 1)))
                else:
                    y_n.append(list(results["tradePrice"][triggerLoc[i]+1:triggerLoc[i+1]]/results["tradePrice"][triggerLoc[i]]))
                    x_n.append(list(range(triggerLoc[i+1] - triggerLoc[i] - 1)))
            
            
            x_p = pd.Series([x for seg in x_p for x in seg])
            x_n = pd.Series([x for seg in x_n for x in seg])
            
            y_p = pd.Series([y/seg[0]-1 for seg in y_p for y in seg])
            y_n = pd.Series([y/seg[0]-1 for seg in y_n for y in seg])
            
            plt.figure()
            plt.scatter(x_p, y_p, marker = ".", s = 5, c = "#00cc00", label = "buy")
            plt.scatter(x_n, y_n, marker = ".", s = 5, c = "#ff0000", label = "sell")
            plt.title("Scatter Plot of time vs returns " + ind)
            plt.legend()
            
            
            avgNeg = y_n.mean()
            avgPos = y_p.mean()
            avgLen = len(results)/len(triggerLoc)
            
            print("average " + ind + " positive return:   " + str(avgPos))
            print("average " + ind + " negative return:   " + str(avgNeg))
            print("average " + ind + " timeframe:         " + str(avgLen))
            print()
            
        return results, triggerLoc
    
    
    
    def coor_RSI(self, tickerList = [], tradeDelay = 0, upperThreshold = 70, lowerThreshold = 30):
        indics = ["RSI"]
        loadedData, trigList = self.loadIndicatorFromDB(tickerList = tickerList, indicators = indics)
        
        loadedData["tradePrice"] = None
        for ind in indics:
            loadedData[self.indicatorList[ind] + "_trig"] = None
        
        for ind in indics:
            indColName = self.indicatorList[ind]
            results = pd.DataFrame(columns = loadedData.columns)
            
            x_p = []
            x_n = []
            y_p = []
            y_n = []
            
            for tick in tickerList:
                print("\rProcessing ticker:  " + str(tick).rjust(6) + "  for " + ind + ".                                       ", end = "")
                
                rslt_df = loadedData.loc[loadedData["ticker_symbol"] == tick]
                
                temp = [1 if (x<lowerThreshold and y>lowerThreshold) else -1 if (x>upperThreshold and y<upperThreshold)
                        else 0 for x,y in zip(rslt_df[indColName][:-1],rslt_df[indColName][1:])]
                
                temp.insert(0,0)
                rslt_df[indColName + "_trig"] = temp
                rslt_df["tradePrice"] = rslt_df["adj_close"].shift(periods = tradeDelay)
                
                results = pd.concat([results, rslt_df])
            
            print("\rPlotting  for  " + ind + ".                                                                      ", end = "")
            triggerLoc = list(np.where(list(results[indColName + "_trig"]))[0])
            
            for i in range(len(triggerLoc)-1):
                if results[indColName + "_trig"][triggerLoc[i]] > 0:
                    y_p.append(list(results["tradePrice"][triggerLoc[i]+1:triggerLoc[i+1]]/results["tradePrice"][triggerLoc[i]]))
                    x_p.append(list(range(triggerLoc[i+1] - triggerLoc[i] - 1)))
                else:
                    y_n.append(list(results["tradePrice"][triggerLoc[i]+1:triggerLoc[i+1]]/results["tradePrice"][triggerLoc[i]]))
                    x_n.append(list(range(triggerLoc[i+1] - triggerLoc[i] - 1)))
            
            
            x_p = pd.Series([x for seg in x_p for x in seg])
            x_n = pd.Series([x for seg in x_n for x in seg])
            
            y_p = pd.Series([y/seg[0]-1 for seg in y_p for y in seg])
            y_n = pd.Series([y/seg[0]-1 for seg in y_n for y in seg])
            
            plt.figure()
            plt.scatter(x_p, y_p, marker = ".", s = 5, c = "#00cc00", label = "buy")
            plt.scatter(x_n, y_n, marker = ".", s = 5, c = "#ff0000", label = "sell")
            plt.title("Scatter Plot of time vs returns " + ind)
            plt.legend()
            
            
            avgNeg = y_n.mean()
            avgPos = y_p.mean()
            avgLen = len(results)/len(triggerLoc)
            
            print("average " + ind + " positive return:   " + str(avgPos))
            print("average " + ind + " negative return:   " + str(avgNeg))
            print("average " + ind + " timeframe:         " + str(avgLen))
            print()
            
        return results, triggerLoc
    
    
    
    def coor_OBV(self, tickerList = [], OBVshift = 1, tradeDelay = 0):
        indics = ["OBV"]
        loadedData, trigList = self.loadIndicatorFromDB(tickerList = tickerList, indicators = indics)
        
        loadedData["tradePrice"] = None
        for ind in indics:
            loadedData[self.indicatorList[ind] + "_trig"] = None
            loadedData[self.indicatorList[ind] + "_comp"] = None
        
        for ind in indics:
            indColName = self.indicatorList[ind]
            results = pd.DataFrame(columns = loadedData.columns)
            
            x_p = []
            x_n = []
            y_p = []
            y_n = []
            
            for tick in tickerList:
                print("\rProcessing ticker:  " + str(tick).rjust(6) + "  for " + ind + ".                                       s", end = "")
                rslt_df = loadedData.loc[loadedData["ticker_symbol"] == tick]
                rslt_df[indColName + "_comp"] = rslt_df[indColName].shift(periods = OBVshift, fill_value = 0)
                rslt_df[indColName + "_trig"] = np.append(np.diff(np.sign(rslt_df[indColName] - rslt_df[indColName + "_comp"])), [0])
                rslt_df["tradePrice"] = rslt_df["adj_close"].shift(periods = tradeDelay)
                
                results = pd.concat([results, rslt_df])
                
            
            print("\rPlotting  for  " + ind + ".                                                                      ", end = "")
            triggerLoc = list(np.where(np.diff(np.sign(rslt_df[indColName] - rslt_df[indColName + "_comp"])))[0])
            
            for i in range(len(triggerLoc)-1):
                if results[indColName + "_trig"][triggerLoc[i]] > 0:
                    y_p.append(list(results["tradePrice"][triggerLoc[i]+1:triggerLoc[i+1]]/results["tradePrice"][triggerLoc[i]]))
                    x_p.append(list(range(triggerLoc[i+1] - triggerLoc[i] - 1)))
                else:
                    y_n.append(list(results["tradePrice"][triggerLoc[i]+1:triggerLoc[i+1]]/results["tradePrice"][triggerLoc[i]]))
                    x_n.append(list(range(triggerLoc[i+1] - triggerLoc[i] - 1)))
            
            
            x_p = pd.Series([x for seg in x_p for x in seg])
            x_n = pd.Series([x for seg in x_n for x in seg])
            
            y_p = pd.Series([y/seg[0]-1 for seg in y_p for y in seg])
            y_n = pd.Series([y/seg[0]-1 for seg in y_n for y in seg])
            
            plt.figure()
            plt.scatter(x_p, y_p, marker = ".", s = 5, c = "#00cc00", label = "buy")
            plt.scatter(x_n, y_n, marker = ".", s = 5, c = "#ff0000", label = "sell")
            plt.title("Scatter Plot of time vs returns " + ind)
            plt.legend()
            
            
            avgNeg = y_n.mean()
            avgPos = y_p.mean()
            avgLen = len(results)/len(triggerLoc)
            
            print("average " + ind + " positive return:   " + str(avgPos))
            print("average " + ind + " negative return:   " + str(avgNeg))
            print("average " + ind + " timeframe:         " + str(avgLen))
            print()
            
        return results, triggerLoc
    
    
    
    def loadIndicatorFromDB(self, tickerList = [], indicators = []):
        self.validate.validateListString(tickerList)
        self.validate.validateListString(indicators)
        for value in indicators:
            if value not in self.indicatorList.keys():
                raise ValueError("Indicators passed are not listed in analysis module.")
        
        results = pd.DataFrame()
        trigList = []
        
        for tick in tickerList:
            print("\rRetrieving ticker:  " + str(tick).rjust(6) + "  and indicators:  " + str(indicators) + ".             ", end = "")
            argList = []
            argList.append(tick)
            queryString = "SELECT [INDICATORS] adj_close, ticker_symbol " +\
                          "FROM daily_adjusted \n" +\
                          "WHERE ticker_symbol = ?;"
            
            # append to the SQL query string and argument list each ticker from the 
            # function inputs
            indicatorString = ""
            for ind in indicators:
                indicatorString += self.indicatorList[ind] + ", "
            
            queryString = queryString.replace("[INDICATORS]", indicatorString)
            
            query = self._cur.execute(queryString, argList)
            cols = [column[0] for column in query.description]
            results = results.append(pd.DataFrame.from_records(data = query.fetchall(), columns = cols), ignore_index=True)
            
            trigList.append(len(results.index))
        
        trigList.pop()
        
        return results, trigList
    
    
    
    def copyTimeSeriesToNewDatabase(self, newDBName, tableName = "daily_adjusted", delCurrentTable = True):
        if self._tickerList == []:
            raise ValueError("Selected Ticker List is empty.  Run 'filterStocksFromDataBase()' to add tickers to the list.")
        if newDBName == "" or not isinstance(newDBName, str) or newDBName == self.mainDBName:
            raise ValueError("Database Name is not valid.")
        
        
        # get the schema from the existing table of time series data
        # https://www.sqlitetutorial.net/sqlite-describe-table/
        schemaText = self._cur.execute("""SELECT sql 
                                          FROM sqlite_schema 
                                          WHERE name = 'daily_adjusted';""").fetchall()
        schemaText = schemaText[0][0] # convert the returned tuple to an executable SQL string command
        schemaText = schemaText[schemaText.find('('):]
        
        
        # create a new database and add a copy of the time series table schema
        self._cur.execute("ATTACH DATABASE ? AS newDB;", [newDBName])
        if delCurrentTable:
            self._cur.execute("DROP TABLE IF EXISTS newDB.daily_adjusted;")
        self._cur.execute("CREATE TABLE newDB.daily_adjusted" + schemaText + ";")
        
        n = 0
        for ticker in self._tickerList:
            n += 1
            print("\rCopying ticker:  " + ticker.rjust(6) + "   (" + str(n).rjust(5) + " of " + str(len(self._tickerList)).ljust(6) + ").", end = "")
            self._cur.execute("""INSERT INTO newDB.daily_adjusted 
                                 SELECT * FROM main.daily_adjusted
                                 WHERE ticker_symbol = ?;""", [ticker])
        
        self.DB.commit()
        self._cur.execute("DETACH DATABASE newDB;")
        print("\nComplete.")
        return
        




if __name__ == "__main__":
    # decomposition = sm.tsa.seasonal_decompose()
    
    # rcParams["figure.figsize"] = 16, 4
    # decomposition.seasonal.plot();
    
    ana = analysis()
    ana.filterStocksFromDataBase(dailyLength = 1250, maxDailyChange = 100, minDailyChange = -80, minDailyVolume = 500000)
    print("Number of stocks selected:  " + str(len(ana._tickerList)) + ".             ")
    
    
    #ana.coor_MACD(tickerList = ana._tickerList)
    ana.coor_BOLL(tickerList = ana._tickerList)
    #ana.coor_MA(tickerList   = ana._tickerList)
    #ana.coor_OBV(tickerList  = ana._tickerList)
    #ana.coor_RSI(tickerList  = ana._tickerList)
    
    
    #ana.copyTimeSeriesToNewDatabase(newDBName = "./DBcopy.db")
    
    
    # vol_diff = [[],[],[]]
    # close_diff = [[],[],[]]
    
    # results = ana.dailyReturns(tickerList=ana._tickerList)
    
    # plt.close('all')
    
    # for ticker in ana._tickerList:
    #     print("\rCalculating volume and price differential stats for:  " + str(ticker).rjust(6) + ".                ", end = "")
    #     try:
    #         stats = ana.compareDataSources(ticker = ticker)
    #         deltaVol = stats["percent_delta_volume"]["yahoo"]
    #         deltaClose = stats["percent_delta_close"]["yahoo"]
            
    #         vol_diff[0].append(max(deltaVol))
    #         vol_diff[1].append(min(deltaVol))
    #         vol_diff[2].append(sum(deltaVol) / len(deltaVol))
            
    #         close_diff[0].append(max(deltaClose))
    #         close_diff[1].append(min(deltaClose))
    #         close_diff[2].append(sum(deltaClose) / len(deltaClose))
            
    #     except missingTicker as mT:
    #         print("\r" + str(mT) + "                                                                     ")
            
    #         vol_diff[0].append("None")
    #         vol_diff[1].append("None")
    #         vol_diff[2].append("None")
            
    #         close_diff[0].append("None")
    #         close_diff[1].append("None")
    #         close_diff[2].append("None")
            
    
    # volAndPriceDiff = np.array(vol_diff + close_diff).T
    # volAndPriceDiff = pd.DataFrame(volAndPriceDiff, columns = ["maxVol", "minVol", "avgVol", "maxPrice", "minPrice", "avgPrice"])    
    # volAndPriceDiff.replace('None', np.nan, inplace=True)
    # cols = volAndPriceDiff.columns
    # volAndPriceDiff[cols] = volAndPriceDiff[cols].apply(pd.to_numeric, errors='coerce')
    # volAndPriceDiff.dropna(axis=0, inplace=True)
    
    
    # print("\rVolume and Price differential calculations Complete.                       ")
    # print("Plotting Figures...                                                ")
    
    
    
    # plt.figure()
    # results["max_daily_change"].hist(bins=100)
    # plt.title("Maximum Daily Change")
    
    
    # plt.figure()
    # results["min_daily_change"].hist(bins=100)
    # plt.title("Minimum Daily Change")

    
    # plt.figure()
    # results["avg_return"].hist(bins=100)
    # plt.title("Average Daily Change")
    
    
    # plt.figure()
    # results["std_return"].hist(bins=100)
    # plt.title("StdDev of Daily Change")
    
    
    # plt.figure()
    # plt.hist(np.log(volAndPriceDiff["maxVol"]/100), bins = 50, label = "Max Vol Diff", histtype='step', stacked=True, fill=False)
    # plt.hist(np.log(volAndPriceDiff["maxPrice"]/100), bins = 50, label = "Max Price Diff", histtype='step', stacked=True, fill=False)
    # plt.title("log(Max)")
    # plt.legend(prop={'size': 10})
    
    
    # plt.figure()
    # plt.hist(np.log(volAndPriceDiff["minVol"].replace(to_replace=0, value=0.000000001)/100), bins = 50, label = "Min Vol Diff", histtype='step', stacked=True, fill=False)
    # plt.hist(np.log(volAndPriceDiff["minPrice"].replace(to_replace=0, value=0.000000001)/100), bins = 50, label = "Min Price Diff", histtype='step', stacked=True, fill=False)
    # plt.title("log(Min)")
    # plt.legend(prop={'size': 10})
    
    
    # plt.figure()
    # plt.hist(np.log(volAndPriceDiff["avgVol"]/100), bins = 50, label = "Avg Vol Diff", histtype='step', stacked=True, fill=False)
    # plt.hist(np.log(volAndPriceDiff["avgPrice"]/100), bins = 50, label = "Avg Price Diff", histtype='step', stacked=True, fill=False)
    # plt.title("log(Avg)")
    # plt.legend(prop={'size': 10})
    
    
    
    
    