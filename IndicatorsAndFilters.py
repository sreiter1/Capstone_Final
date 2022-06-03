# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:16:33 2021

@author: sreit
"""


import pandas as pd
import datetime
import sqlite3
import warnings
import commonUtilities





class filterData:
    def __init__(self, dataBaseSaveFile = "./SQLiteDB/stockData.db", dataBaseThreadCheck = True):
        self.DB = sqlite3.connect(dataBaseSaveFile, check_same_thread = dataBaseThreadCheck)
        self._cur = self.DB.cursor()
        self._tickerList = []     # Empty list that gets filled with a list of tickers to be considered
        
        self.validate = commonUtilities.validationFunctions()
        
        # Converts a user string to the names of tables, and for 'ticker_symbol_list'
        self._tickerConversionTable = commonUtilities.conversionTables.tickerConversionTable
        
        # Converts user unput to the columns in the table.  Provides a filter to 
        # prevent database corruption.
        self._dailyConversionTable = commonUtilities.conversionTables.dailyConversionTable
        self.dailyTableNames = ["alpha", "yahoo"]
        
        
        
    
    def checkTickersInDatabase(self, 
                              returnError = False, 
                              returnDate = False,
                              returnNumRecord= False, 
                              tickerList = []):
        
        # This function confirms that tickers that are in the self._tickerList
        # are also in the 'ticker_symbol_list'.  Can optionally return the other 
        # data contained in the table as a pandas dataframe.
        
        # check function inputs for problematic values
        self.validate.validateBool(returnError)
        self.validate.validateBool(returnDate)
        self.validate.validateBool(returnNumRecord)
        self.validate.validateListString(tickerList)
        
        
        tickerGroups = int(len(tickerList) / 100) + 1
        results = pd.DataFrame()
        
        for i in range(tickerGroups):
            minIndex = i * 100
            maxIndex = minIndex + 100 if (minIndex + 100 < len(tickerList)) else len(tickerList) - 1
            
            tickerListSubset = tickerList[minIndex:maxIndex]
            
            # start a SQL query string and an list of arguments.  Leave a space for
            # the specific columns that should be requested ("[REQUEST_WHAT]")
            queryString = "SELECT ticker_symbol  [RETURN_WHAT]" +\
                          "FROM ticker_symbol_list \n" +\
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
            
            # Create a list of what data should be checked, and append column names 
            # for that data to the list
            returnWhat = []
            
            if returnError:
                returnWhat.append("error_fundamental_overview")
                returnWhat.append("error_income_statement")
                returnWhat.append("error_earnings")
                returnWhat.append("error_alpha_daily")
                returnWhat.append("error_balance_sheet")
                returnWhat.append("error_cash_flow")
            if returnDate:
                returnWhat.append("date_fundamental_overview")
                returnWhat.append("date_income_statement")
                returnWhat.append("date_earnings")
                returnWhat.append("date_alpha_daily")
                returnWhat.append("date_balance_sheet")
                returnWhat.append("date_cash_flow")
            if returnNumRecord:
                returnWhat.append("records_fundamental_overview")
                returnWhat.append("records_income_statement")
                returnWhat.append("records_earnings")
                returnWhat.append("records_alpha_daily")
                returnWhat.append("records_balance_sheet")
                returnWhat.append("records_cash_flow")
            
            
            # convert the list of column names to a string and place it into the SQL query
            if returnWhat != []:
                returnWhat = "".join(", " + cond for cond in returnWhat)
                returnWhat += " "
            else:
                returnWhat = " "
            
            queryString = queryString.replace(" [RETURN_WHAT]", returnWhat)
            
            # execute the SQL query and convert the response to a pandas dataframe
            query = self._cur.execute(queryString, argList)
            cols = [column[0] for column in query.description]
            results = pd.concat([results, pd.DataFrame.from_records(data = query.fetchall(), columns = cols)], axis=0)
        
        # make the list of tickers in the dataframe available outside the funciton
        self._tickerList = results["ticker_symbol"].tolist()
        
        # return the dataframe of data
        return results
        
        
    
    def autoSaveIndicators(self,
                           indicators = None,
                           priceData = "ADJCLOSE",
                           useAdjVolume = False, 
                           tickerList = [],
                           source = None):
        
        # Function calculates the indicators and saves them to the database.
        # Indicators include things like RSI, OBV, moving averages, etc.
        # Works on tickers contained in self._tickerList, for the indicators
        # passed to the function, and on the price passed to the function.  
        # Defaults to the adjusted close price and all the indicators.
        
        if tickerList == []:
            tickerList = self._tickerList
        if tickerList == []:
            queryString  = "SELECT ticker_symbol "
            queryString += " FROM ticker_symbol_list \n"
            queryString += " WHERE records_alpha_daily >= 1 \n"
            query = self._cur.execute(queryString)
            cols = [column[0] for column in query.description]
            results = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
            self._tickerList = list(results["ticker_symbol"])
            self._tickerList.sort()
            tickerList = self._tickerList
            del results
            
        if source == None or source not in self.dailyTableNames:
            source = self.dailyTableNames[0]
        assert isinstance(source, str), "Error, 'source' in 'autoSaveIndicators()' is not a string."
        
        self.validate.validateListString(tickerList)
        if isinstance(indicators, str):
            indicators = ["".join(e for e in indicators if e.isalpha())] # verify that all the text is alphabetical
        if indicators == None:
            indicators = ["ADJRATIO", "MA20", "MA50", "BOLLINGER20", "BOLLINGER50", \
                          "MACD12", "MACD19", "VOL20", "VOL50", "OBV", "DAYCHANGE", \
                          "TOTALCHANGE", "RSI", "IDEAL"]
        
        # esnure that the related "Typical Price" is included when the bollinger 
        # bands are calculated.  NOTE: TP doesn't get a specific calc; it is 
        # only calculated with Bollinger selected.
        if "BOLLINGER20" in indicators:
            indicators.insert(indicators.index("BOLLINGER20"), "TP20")
        if "BOLLINGER50" in indicators:
            indicators.insert(indicators.index("BOLLINGER50"), "TP50")
        if "IDEAL" in indicators:
            indicators.insert(indicators.index("IDEAL"), "IDEAL_TRIG")
            indicators.insert(indicators.index("IDEAL"), "IDEAL_LOW")
            indicators.insert(indicators.index("IDEAL"), "IDEAL_HIGH")
            
        
        # Verify the tickers that are being searched are actually in the database.
        self.checkTickersInDatabase(tickerList = tickerList)
        
        # Request the data for each ticker in turn
        print("Calculating indicators:   ", end="")
        for i in indicators[:-1]:
            print(i + ", ", end = "")
        print("and " + indicators[-1] + ".                                   \n")
        
        data = pd.DataFrame()
        stockCount = 0
        for ticker in tickerList:
            stockCount += 1
            if ticker != ticker.upper():
                print("\r" + ticker + "  Not Upper.")
            if ticker in ["CODE", "Code", "code"]:
                print("\r             " + ticker)
            
            print("\rComputing daily indicators for  " + str(ticker).rjust(7) + "      (" + str(stockCount).rjust(6) + " of " + str(len(tickerList)).ljust(6) + ")         ", end = "")
            
            queryString  = "SELECT * "
            queryString += "FROM " + source + "_daily \n"
            queryString += "WHERE ticker_symbol = '" + ticker + "';\n"
            
            query = self._cur.execute(queryString)
            cols = [column[0] for column in query.description]
            data = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
            
            # convert the recordDate from a string to a datetime object, and set as the index
            data["recordDate"] = pd.to_datetime(data["recordDate"])
            data = data.set_index("recordDate")
            data.sort_index()
            
            # Copy data from results to lists
            priceHist = list(data[self._dailyConversionTable[priceData]])
            closeHist = list(data[self._dailyConversionTable["CLOSE"]])
            adjCloseHist = list(data[self._dailyConversionTable["ADJCLOSE"]])
            volHist = list(data[self._dailyConversionTable["VOLUME"]])
            
            if useAdjVolume:
                volHist = [v * c / a for v, c, a in zip(volHist, closeHist, adjCloseHist)]
            
            for indicator in indicators:
                if indicator == "MACD12":
                    series = pd.Series(self._macd(hist = priceHist, period_1 = 12, period_2 = 26, periodSignal = 9))
                if indicator == "MACD19":
                    series = pd.Series(self._macd(hist = priceHist, period_1 = 19, period_2 = 39, periodSignal = 9))
                if indicator == "MA20":
                    series = pd.Series(self._simpleMovingAverage(hist = priceHist, periods = 20))
                if indicator == "MA50":
                    series = pd.Series(self._simpleMovingAverage(hist = priceHist, periods = 50))
                if indicator == "VOL20":
                    series = pd.Series(self._simpleMovingAverage(hist = volHist, periods = 20))
                if indicator == "VOL50":
                    series = pd.Series(self._simpleMovingAverage(hist = volHist, periods = 50))
                if indicator == "RSI":
                    series = pd.Series(self._rsi(hist = priceHist, exp = True))
                if indicator == "DAYCHANGE":
                    series = pd.Series(self._percentChangeDaily(hist = priceHist))
                if indicator == "TOTALCHANGE":
                    series = pd.Series(self._percentChangeTotal(hist = priceHist))
                if indicator == "ADJRATIO":
                    series = pd.Series(self._stockAdjustRatio(close = closeHist, adjClose = adjCloseHist))
                if indicator == "OBV":
                    series = pd.Series(self._onBalanceVolume(close = closeHist, volume = volHist))
                if indicator == "IDEAL":
                    ret, high, low, trig  = pd.Series(self._tradeIdealReturn(lookAheadDays = 15,
                                                                             adjRatIn  = list(data[self._dailyConversionTable["ADJRATIO"]]) , 
                                                                             low_hist  = list(data[self._dailyConversionTable["CLOSE"   ]]) ,
                                                                             high_hist = list(data[self._dailyConversionTable["CLOSE"   ]]) ))
                    
                    data[self._dailyConversionTable["IDEAL"]]      = ret
                    data[self._dailyConversionTable["IDEAL_HIGH"]] = high
                    data[self._dailyConversionTable["IDEAL_LOW"]]  = low
                    data[self._dailyConversionTable["IDEAL_TRIG"]] = trig
                    continue
                
                
                if indicator == "BOLLINGER20":
                    TP, bollinger = pd.Series(self._bollingerBands(high = list(data[self._dailyConversionTable["HIGH"]]), 
                                                                   low = list(data[self._dailyConversionTable["LOW"]]), 
                                                                   close = closeHist, 
                                                                   adjClose = adjCloseHist, 
                                                                   periods = 20))
                    data[self._dailyConversionTable["TP20"]] = TP
                    data[self._dailyConversionTable["BOLLINGER20"]] = bollinger
                    continue
                
                
                if indicator == "BOLLINGER50":
                    TP, bollinger = pd.Series(self._bollingerBands(high = list(data[self._dailyConversionTable["HIGH"]]), 
                                                                   low = list(data[self._dailyConversionTable["LOW"]]), 
                                                                   close = closeHist, 
                                                                   adjClose = adjCloseHist, 
                                                                   periods = 50))
                    data[self._dailyConversionTable["TP20"]] = TP
                    data[self._dailyConversionTable["BOLLINGER50"]] = bollinger
                    continue
                
                
                data[self._dailyConversionTable[indicator]] = series.values
            
            data = data.set_index(data.index)
            self._saveIndicatorsToDB(ticker_symbol = ticker,
                                     indicators = indicators,
                                     dataframe = data,
                                     useExtras = True)
        
        print()
        return data
    
    
    
    def _saveIndicatorsToDB(self, ticker_symbol, indicators, dataframe, useExtras = False):
        # Saves the data extracted from the price history (i.e. moving averages)
        # back to the database.  
        
        # Ensure the record exists in the database by either inserting or ignoring the unique constraint
        sqlString  = "INSERT OR IGNORE INTO daily_adjusted (ticker_symbol, recordDate) \n"
        sqlString += "VALUES(?,?);"
        
        sqlArray  = pd.DataFrame()
        sqlArray["ticker_symbol"] = dataframe["ticker_symbol"]
        sqlArray["recordDate"]    = dataframe.index.astype(str)
        sqlArray = sqlArray.values.tolist()
        
        self._cur.executemany(sqlString, sqlArray)
        self.DB.commit()
        
        # remove this ariable to ensure it does not impact the update lines
        del sqlArray
        
        
        extras = ["OPEN", "HIGH", "LOW", "CLOSE", "ADJCLOSE", "VOLUME", "DIVIDEND", "SPLIT"]
        
        
        # start SQL query line for updating the indicators
        sqlString  = "UPDATE daily_adjusted \n SET "
        sqlArray   = pd.DataFrame()
        
        # add each indicator in the list of indicators to the SQL query, along
        # with the associated values.
        
        if useExtras:
            for extra in extras:
                sqlString += str(self._dailyConversionTable[extra]) + " = ?, \n     "
                sqlArray[self._dailyConversionTable[extra]] = dataframe[self._dailyConversionTable[extra]]
            
        for indicator in indicators:
            sqlString += str(self._dailyConversionTable[indicator]) + " = ?, \n     "
            sqlArray[self._dailyConversionTable[indicator]] = dataframe[self._dailyConversionTable[indicator]]
            
        
        # finish the string, execute the SQL transaction, and commit the changes
        sqlString  = sqlString[:-8] + "\n"
        sqlString += "WHERE ticker_symbol = ? AND recordDate = ?; \n"
        sqlArray["ticker_symbol"] = ticker_symbol
        sqlArray["recordDate"] = dataframe.index.astype(str)
        sqlArray = sqlArray.values.tolist()
        
        self._cur.executemany(sqlString, sqlArray)
        self.DB.commit()
    
    
    
    def populateSummaryData(self):
        # the database has a table called "summary_data" that contains some 
        # data from several tables in one location to make certain queries 
        # easier and faster.  This function ensures that all the tickers in 
        # "ticker_symbol_list" are contained in the "summary_data"
        
        # get all the tickers in the database and compile them into a list
        query  = "SELECT ticker_symbol FROM ticker_symbol_list;\n"
        result = self._cur.execute(query)
        data = result.fetchall()
        tickerList = [item for t in data for item in t]
        
        # count the number of tickers for a command line update; set to 0 initially.
        populationCount = 0
        
        # for each ticker, increase the count by one, add the ticker to the summary_data,
        # and make the calculated date blank
        for ticker in tickerList:
            populationCount += 1
            print("\rAdding ticker   '" + str(ticker).rjust(7) + "'      (" + str(populationCount).rjust(6) + " of " + str(len(tickerList)).ljust(6) + ")    to summary statistics     ", end = "")
            query  = "INSERT OR IGNORE INTO summary_data (ticker_symbol, date_calculated)\n"
            query += "VALUES(?, ?)"
            argList = [ticker, ""]
            self._cur.execute(query, argList)
        
        # commit the changes to the database
        self.DB.commit()
        print()
        
        # call the function that fills the "summary_data" table in the database
        self._calculateSummaryData()
        
        
        
    def _calculateSummaryData(self, lastCalculated = -1):
        # the database has a table called "summary_data" that contains some 
        # data from several tables in one location to make certain queries 
        # easier and faster.  This function ensures that all the tickers in 
        # "ticker_symbol_list" are contained in the "summary_data"
        
        # get today's date as a string
        dateCalculated = str(datetime.datetime.now().date())
        
        # check function inputs for error / unexpected data.  Also sets 
        # "lastCalculated" based on the user input to be the date where 
        # anything with a refresh date older than this will get refreshed.  
        # Other records will be ignored.
        if isinstance(lastCalculated, str):
            self.validate.validateDateString(lastCalculated)
            lastCalculated = datetime.datetime.strptime(lastCalculated, '%Y-%m-%d').date()
        elif isinstance(lastCalculated, int):
            lastCalculated = datetime.date.today()-datetime.timedelta(days = lastCalculated)
        
        # get a list of all the tickers in the "summary_data" table from the database
        query  = "SELECT ticker_symbol, date_calculated FROM summary_data;\n"
        result = self._cur.execute(query)
        data = result.fetchall()
        
        # convert the response to a list of lists, formatted such that each record 
        # is an entry in the outer list, while the inner list contains the ticker
        # and the calculated date
        summaryData = [list(t) for t in data]  
        
        # counter and total variables for a command line progress update
        length_SummaryData = len(summaryData)
        count_records = 0
        
        # 
        for record in summaryData:
            # progress update
            count_records += 1
            print("\rCalculating summary data  for  " + str(record[0]).rjust(7) + "    (" + str(count_records).rjust(6) + " of " + str(length_SummaryData).ljust(6) + ")             ", end = "")
            
            # the date is saved as a string in the database, so it needs to be 
            # converted back to a date-time object.  This function sets that 
            # datetime object as the date from the database or 01 Jan 1900.
            # This object is assigned to "calcDate"
            if (record[1] != ""):
                calcDate = datetime.datetime.strptime(record[1], '%Y-%m-%d').date()
            else:
                calcDate = datetime.datetime(1900, 1, 1).date()
            
            # check to see if the last time the data was refreshed is older than
            # the user input date, and if it is older, make updates to "summary_data"
            if calcDate < lastCalculated:
                # SQL query to get data from the "fundamental_overview" table,
                # convert it to a pandas dataframe, and assign it to fundData.
                # Then extract the relevant data from the dataframe and assign 
                # it to a specific variable.
                queryString  = "SELECT recordDate, country, exchange, sector, industry FROM fundamental_overview \n"
                queryString += "WHERE ticker_symbol = ?;\n"
                
                query = self._cur.execute(queryString, [record[0]])
                cols = [column[0] for column in query.description]
                fundData = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
                fundData = fundData.set_index("recordDate")
                fundData = fundData.sort_index()
                if len(fundData.index) > 0:
                    country = fundData["country"][-1]
                    exchange = fundData["exchange"][-1]
                    sector = fundData["sector"][-1]
                    industry = fundData["industry"][-1]
                else:
                    country = "None"
                    exchange = "None"
                    sector = "None"
                    industry = "None"
                
                # SQL query to get data from the "income_statement" table,
                # convert it to a pandas dataframe, and assign it to incData.
                # Then extract the relevant data from the dataframe and assign 
                # it to a specific variable.
                queryString  = "SELECT recordDate, netIncome, totalRevenue FROM income_statement \n"
                queryString += "WHERE ticker_symbol = ? AND annualReport = ?;\n"
                
                query = self._cur.execute(queryString, [record[0], "0"])
                cols = [column[0] for column in query.description]
                incData = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
                incData = incData.set_index("recordDate")
                incData = incData.sort_index()
                try:
                    profitMargin = float(incData["netIncome"][-1]) / float(incData["totalRevenue"][-1])
                except:
                    profitMargin = "None"
                
                
                
                # SQL query to get data from the "balance_sheet" table,
                # convert it to a pandas dataframe, and assign it to balData.
                # Then extract the relevant data from the dataframe and assign 
                # it to a specific variable.
                queryString  = "SELECT recordDate, totalShareholderEquity, commonStockSharesOutstanding FROM balance_sheet \n"
                queryString += "WHERE ticker_symbol = ? AND annualReport = ?;\n"
                
                query = self._cur.execute(queryString, [record[0], "0"])
                cols = [column[0] for column in query.description]
                balData = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
                balData = balData.set_index("recordDate")
                balData = balData.sort_index()
                try:
                    shareHolderEquity = float(balData["totalShareholderEquity"][-1])
                except:
                    shareHolderEquity = "None"
                try:
                    commonShares = balData["commonStockSharesOutstanding"][-1].astype(float)
                except:
                    commonShares = "None"
                
                
                
                # SQL query to get data from the "earnings" table,
                # convert it to a pandas dataframe, and assign it to earnData.
                # Then extract the relevant data from the dataframe and assign 
                # it to a specific variable.
                queryString  = "SELECT recordDate, reportedEPS FROM earnings \n"
                queryString += "WHERE ticker_symbol = ?;\n"
                
                query = self._cur.execute(queryString, [record[0]])
                cols = [column[0] for column in query.description]
                earnData = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
                earnData = earnData.set_index("recordDate")
                earnData = earnData.sort_index()
                try:
                    EPS = float(earnData["reportedEPS"][-1])
                except:
                    EPS = "None"
                
                
                
                # SQL query to get data from the "daily_adjusted" table,
                # convert it to a pandas dataframe, and assign it to dayData.
                # Then extract the relevant data from the dataframe and assign 
                # it to a specific variable.
                queryString  = "SELECT recordDate, adj_close, percent_cng_day, percent_cng_tot, volume FROM daily_adjusted \n"
                queryString += "WHERE ticker_symbol = ?;\n"
                
                query = self._cur.execute(queryString, [record[0]])
                cols = [column[0] for column in query.description]
                dayData = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
                dayData = dayData.set_index("recordDate")
                dayData = dayData.sort_index()
                if len(dayData.index) > 0:
                    dailyLength = len(dayData.index)
                    currentPrice = float(dayData["adj_close"][-1])
                else:
                    dailyLength = 0
                    currentPrice = "None"
                
                try:
                    marketCap = currentPrice * commonShares
                except:
                    marketCap = "None"
                    
                try:
                    peRatio = currentPrice / EPS
                except:
                    peRatio = "None"
                
                
                try:
                    avgReturn = dayData["percent_cng_day"].mean()
                    stdReturn = dayData["percent_cng_day"].std()
                    maxReturn = dayData["percent_cng_day"].max()
                    minReturn = dayData["percent_cng_day"].min()
                except:
                    avgReturn = "None"
                    stdReturn = "None"
                    maxReturn = "None"
                    minReturn = "None"
                
                try:
                    compReturn = dayData["percent_cng_tot"].mean()
                    compStdDev = dayData["percent_cng_tot"].std()
                except:
                    compReturn = "None"
                    compStdDev = "None"
                    
                try:
                    minVol = dayData["volume"].min()
                except:
                    minVol = "None"
                    
                
                # Create a SQL update command to assign the values collected above
                # and assign them to specific columns in "summary_data".  
                queryString  = "UPDATE summary_data \n"
                queryString += "SET date_calculated = ?,"
                queryString += "    daily_length = ?,"
                queryString += "    country = ?,"
                queryString += "    exchange = ?,"
                queryString += "    sector = ?,"
                queryString += "    industry = ?,"
                queryString += "    earnings_per_share = ?,"
                queryString += "    profit_margin = ?,"
                queryString += "    share_holder_equity = ?,"
                queryString += "    common_shares_outstanding = ?,"
                queryString += "    current_price = ?,"
                queryString += "    market_capitalization = ?,"
                queryString += "    pe_ratio = ?,"
                queryString += "    avg_return = ?,"
                queryString += "    std_return = ?,"
                queryString += "    max_daily_change = ?,"
                queryString += "    min_daily_change = ?,"
                queryString += "    min_daily_volume = ?,"
                queryString += "    comp_return = ?,"
                queryString += "    comp_stddev = ?"
                queryString += " WHERE ticker_symbol = ?"
                
                argList = [dateCalculated, 
                           dailyLength, 
                           country,
                           exchange, 
                           sector, 
                           industry, 
                           EPS,
                           profitMargin, 
                           shareHolderEquity, 
                           commonShares,
                           currentPrice, 
                           marketCap, 
                           peRatio,
                           avgReturn,
                           stdReturn,
                           maxReturn,
                           minReturn,
                           minVol,
                           compReturn,
                           compStdDev,
                           record[0]]
                
                
                # Execute and commit the SQL update
                self._cur.execute(queryString, argList)
                self.DB.commit()
        
        print()
    
    

    def _macdExp(self, 
                 hist, 
                 period_1 = 12, 
                 period_2 = 26, 
                 periodSignal = 9):
        # calculate the moving average convergence-divergence using an exponential moving average
        
        self.validate.validateListFloat(hist)
        self.validate.validateInteger(period_1)
        self.validate.validateInteger(period_2)
        self.validate.validateInteger(periodSignal)
        
        assert period_1 >= 2, "'period_1' must be an integer of at least 2."
        assert period_2 >= 2, "'period_2' must be an integer of at least 2."
        assert periodSignal >= 2, "'periodSignal' must be an integer of at least 2."
        assert not(period_1 == period_2), "Periods should not be identical for MACD calculation"
            
        # swap the MACD period if the first is larger than the second
        if period_1 < period_2:
            temp = period_1
            period_1 = period_2
            period_2 = temp
            warnings.warn("Period 1 should be smaller than Period 2 in MACD calculation; swapping values.")
        
        
        # calculate the moving averages based on the input number of periods, 
        # and calculate the MACD based on the difference between the moving averages
        ma1 = self._expMovingAvg(hist = hist, periods = period_1)
        ma2 = self._expMovingAvg(hist = hist, periods = period_2)
        macd = [(a - b) for a, b in zip(ma1, ma2)]
        
        # calulate the moving average signal line
        signal = self._expMovingAvg(hist = macd, periods = periodSignal)
        
        # calculate the MACD histogram (positive is upward momentum, negative
        # is downward momentum).
        result = [(ma - sig) for ma, sig in zip(macd, signal)]
        
        # return the list of histogram values.  Not regularised in any way, nor 
        # does the result contain the actual moving average data.
        return result
    
    
    
    def _macd(self, 
              hist, 
              period_1 = 12, 
              period_2 = 26, 
              periodSignal = 9):
        # calculate the moving average convergence-divergence using a simple moving average
        
        self.validate.validateListFloat(hist)
        self.validate.validateInteger(period_1)
        self.validate.validateInteger(period_2)
        self.validate.validateInteger(periodSignal)
        
        assert period_1 >= 2, "'period_1' must be an integer of at least 2."
        assert period_2 >= 2, "'period_2' must be an integer of at least 2."
        assert periodSignal >= 2, "'periodSignal' must be an integer of at least 2."
        assert not(period_1 == period_2), "Periods should not be identical for MACD calculation"
        
        # swap the MACD period if the first is larger than the second
        if period_1 > period_2:
            temp = period_1
            period_1 = period_2
            period_2 = temp
            warnings.warn("Period 1 should be smaller than Period 2 in MACD calculation; swapping values.")
        
        # calculate the moving averages based on the input number of periods, 
        # and calculate the MACD based on the difference between the moving averages
        ma1 = self._simpleMovingAverage(hist = hist, periods = period_1)
        ma2 = self._simpleMovingAverage(hist = hist, periods = period_2)
        macd = [(a - b) for a, b in zip(ma1, ma2)]
        
        # calulate the moving average signal line
        signal = self._simpleMovingAverage(hist = macd, periods = periodSignal)
        
        # calculate the MACD histogram (positive is upward momentum, negative
        # is downward momentum).  
        result = [(ma - sig) for ma, sig in zip(macd, signal)]
        
        # return the list of histogram values.  Not regularised in any way, nor 
        # does the result contain the actual moving average data.
        return result
    


    def _expMovingAvg(self, 
                      hist, 
                      periods = 20, 
                      smoothing = 2):
        # calculate the exponential moving average 
        self.validate.validateListFloat(hist)
        self.validate.validateInteger(periods)
        assert periods >= 2, "'periods' must be an integer of at least 2."
        
        # initiate the geometric scaling factor for the exponential moving average
        # and set the initial moving average to be the first value in the price 
        # history
        multiplier = smoothing / (periods + 1)
        ema = [hist[0]]
        std = [0]
        
        # for each value after the first, calculate the expoential moving average 
        # and append it to the list of moving average values
        for val in hist[1:]:
            nextEma = (ema[-1] * (1 - multiplier) +  val * multiplier)
            ema.append(nextEma)
            std.append(sum(((val - ema[-1]) ** 2) ) / periods)
        
        # return the list of the moving average values
        return ema



    def _simpleMovingAverage(self, 
                             hist, 
                             periods = 20):
        # calculate the simple moving average
        self.validate.validateListFloat(hist)
        self.validate.validateInteger(periods)
        assert periods >= 2, "'periods' must be an integer of at least 2."
        
        # create a list of the historical price data where the length is 
        # the number of periods, fill it with the first price point, and 
        # set the moving average as the start the moving average as the first 
        # value in the price data list.
        maArray = [hist[0] for i in range(periods)]  # moving average window
        sma = [hist[0]]                              # moving average
        
        # for each value in the price history, drop the first (oldest) value in 
        # the moving average window, append the next value from the price history
        # to the moving average window, and calculate/append the moving average
        # to the moving average list.
        for val in hist[1:]:
            maArray.pop(0)
            maArray.append(val)
            sma.append(sum(maArray) / periods)
        
        # return the list of the moving average
        return sma
    
    
    
    def _bollingerBands(self,
                        high, 
                        low, 
                        close, 
                        adjClose, 
                        periods = 20):
    
        # calculate the simple moving average
        self.validate.validateListFloat(high)
        self.validate.validateListFloat(low)
        self.validate.validateListFloat(close)
        self.validate.validateListFloat(adjClose)
        self.validate.validateInteger(periods)
        assert periods >= 2, "'periods' must be an integer of at least 2."
        
        # create a list of the historical price data where the length is 
        # the number of periods, fill it with the first price point, and 
        # set the Typical Price as the start the average of the high, low, and 
        # close (adjusted by adjclose).
        
        TP = [(a / c) * (h + l + c) / 3 for a, h, l, c in zip(adjClose, high, low, close)]
        
        maArray = [TP[0] for i in range(periods)]    # moving average window
        sma = [TP[0]]                                # moving average
        std = [0]                                    # moving standard deviation
        
        # for each value in the price history, drop the first (oldest) value in 
        # the moving average window, append the next value from the price history
        # to the moving average window, and calculate/append the moving average
        # to the moving average list.
        for val in TP[1:]:
            maArray.pop(0)
            maArray.append(val)
            sma.append(sum(maArray) / periods)
            std.append((sum([((ma - sma[-1]) ** 2) for ma in maArray]) / periods) ** 0.5)
                    
        # return the list of the moving average
        return sma, std



    def _rsi(self, 
             hist, 
             periods = 14,
             exp = False):
        # calculate the Relative Strength Indicator (RSI); not matching other dataset at barchart
        
        # check inputs for errors or problems
        self.validate.validateListFloat(hist)
        self.validate.validateInteger(periods)
        assert periods >= 2, "'periods' must be an integer of at least 2."
        try:
            assert len(hist) > periods+1, "Length of the history provided to RSI insufficient to meet period requirement."
        except:
            rsiArray = [0 for i in range(len(hist))]   # fills the rsi values with 0 when there is insufficient data
            return rsiArray
        
        
        # Calculate the percent change from one day to the next, and inert a '0' as 
        # the first value. 
        changeArray = [(b-a) for a,b in zip(hist[:-1], hist[1:])]
        changeArray.insert(0,0)
        
        # create a list of the gains (percent change > 0), and losses (percent 
        # change < 0).  The lists will correspond element-for-element with the
        # percent change array, and will have 0's to pad those elements that 
        # should not be in that list.
        gainArray = [abs(a) if a > 0 else 0 for a in changeArray]
        lossArray = [abs(a) if a < 0 else 0 for a in changeArray]
        
        # create a list of rsi values with the length = number of periods, and 
        # will it with 0's.
        rsiArray = [0 for i in range(periods)]
        
        # exp == true when the RSI should use the exponential moving average, and 
        # false when using the simple moving average.
        if exp:
            # get an average for the gain and loss in the price history for the 
            # first number of periods
            incrementalGain = sum(gainArray[:periods]) / periods
            incrementalLoss = sum(lossArray[:periods]) / periods
            
            # For each element in the price history from "periods" (where the RSI 
            # becomes valid) through the end of the price history, calculate the
            # exponential moving average of the gain and of the loss, calculate 
            # the RSI based on those values, and set RS = 0 if the incremental 
            # loss would cause a divide-by-zero error
            for i in range(periods, len(hist)):
                incrementalGain = ((incrementalGain * (periods-1)) + gainArray[i]) / periods
                incrementalLoss = max(   ((incrementalLoss * (periods-1)) + lossArray[i]) / periods,    0.0000000001)
                
                rs = incrementalGain / incrementalLoss
                
                # calcuate rsi from the rs value and append it to the array
                rsi = 100 - (100 / (1 + rs))
                rsiArray.append(rsi)
            
        
        else:
            # For each element in the price history from "periods" (where the RSI 
            # becomes valid) through the end of the price history, calculate the
            # simple moving average of the gain and of the loss, calculate 
            # the RSI based on those values, and set RS = 0 if the incremental 
            # loss would cause a divide-by-zero error
            
            
            # get an average for the gain and loss in the price history for the 
            # first number of periods
            incrementalGain = sum(gainArray[:periods]) / periods
            incrementalLoss = sum(lossArray[:periods]) / periods
            
            
            for i in range(periods, len(hist)):
                incrementalGain = sum(gainArray[i-periods:i]) / periods
                incrementalLoss = max(    sum(lossArray[i-periods:i]) / periods,    0.000001)
                
                rs = incrementalGain / incrementalLoss
                
                # calcuate rsi from the rs value and append it to the array
                rsi = 100 - (100 / (1 + rs))
                rsiArray.append(rsi)
        
        # return the array of RSI values
        return rsiArray



    def _percentChangeDaily(self, hist):
        # calculate the percent change between days
        
        # check for unexpected inputs
        self.validate.validateListFloat(hist)
        
        # calculate the percent change as (next - prev) / prev, and then insert
        # 0 as the first value
        percentChange = [100 * (b-a)/a for a,b in zip(hist[:-1], hist[1:])]
        percentChange.insert(0, 0)
        
        return percentChange
    
    
    
    def _percentChangeTotal(self, hist):
        # calculate the percent change from the start of the data
        
        # check for unexpected inputs
        self.validate.validateListFloat(hist)
        
        # calculate the percent change as (next - prev) / prev, and then insert
        # 0 as the first value
        percentChange = [100 * (a-hist[0])/hist[0] for a in hist[1:]]
        percentChange.insert(0, 0)
        
        return percentChange

        

    def _stockAdjustRatio(self, 
                          close, 
                          adjClose):
        # calculate the total adjustment ratio between the close and adjusted close
        # prices.  
        self.validate.validateListFloat(close)
        self.validate.validateListFloat(adjClose)
        assert len(close) == len(adjClose), "'closePrice' and 'adjClose' must have the same length.  "
            
        adjustRatio = [c / a for c, a in zip(close, adjClose)]
        return adjustRatio
    
    
    
    def _onBalanceVolume(self, 
                         close, 
                         volume):
        # calculates the on-balance volume indicator
        self.validate.validateListFloat(close)
        self.validate.validateListFloat(volume)
        assert len(close) == len(volume), "'closePrice' and 'volume' must have the same length."
        
        
        # first value is 0
        OBV = [0]            
        
        # then each subsequent OBV is the previous OBV +/- the present volume
        # based on whether the price change was positive or negative
        for i in range(1,len(close)):
            if close[i] - close[i-1] > 0:
                OBV.append(OBV[-1] + volume[i])
            elif close[i] - close[i-1] == 0:
                OBV.append(OBV[-1] + 0)
            elif close[i] - close[i-1] < 0:
                OBV.append(OBV[-1] - volume[i])
        
        return OBV
    
    
    
    
    def _tradeIdealHigh(self, 
                         lookAhead = 15,  # number of days to look forward
                         hist = []):
        
        lenHist = len(hist)
            
        if lenHist <= 1:
            return [0]
        
        ideal_h = [max(max(hist[i:min(i+lookAhead,lenHist)]), 0.0001) for i in range(lenHist-1)]  # gets highest price between tomorrow and lookAhead days from now
        ideal_h.append(hist[-1])
        
        return ideal_h
    
    
    
    
    def _tradeIdealLow(self, 
                        lookAhead = 15,  # number of days to look forward
                        hist = []):
        
        lenHist = len(hist)
        
        if lenHist <= 1:
            return [0]
        
        ideal_l = [max(min(hist[i:min(i+lookAhead,lenHist)]), 0.0001) for i in range(lenHist-1)]  # gets lowest price between tomorrow and lookAhead days from now
        ideal_l.append(hist[-1])
        
        return ideal_l
    
    
    
    
    def _tradeIdealReturn(self, 
                          lookAheadDays = 15,
                          adjRatIn  = [], 
                          low_hist  = [],
                          high_hist = []):
        
        
        if adjRatIn == [] and high_hist != []:
            adjRatIn = [1] * len(high_hist)
            
        high_hist = [h/a for h,a in zip(high_hist, adjRatIn)]
        low_hist  = [l/a for l,a in zip(low_hist,  adjRatIn)]
        
        
        high_price = self._tradeIdealHigh(lookAhead = lookAheadDays, hist = high_hist)
        low_price  = self._tradeIdealLow( lookAhead = lookAheadDays, hist =  low_hist)
        ret     = [1]*len(high_price)
        trig    = [0]*len(high_price)
        
        
        buyState = 0  # own no stocks
        dollars  = 1  # have 1 dollar
        shares   = 0  # have 0 stocks
        
        for i in range(len(high_price)):
            if low_price[i] == low_hist[i] and (buyState == 0 or buyState == -1):
                shares = dollars / low_price[i]
                dollars = 0
                buyState = 1
                
            elif high_price[i] == high_hist[i] and buyState == 1:
                dollars = shares * high_price[i]
                shares = 0
                buyState = -1
                trig[i] = -1
                
            ret[i]  = max(dollars, shares*(low_hist[i] + high_hist[i])/2)
            trig[i] = buyState
        
        
        return ret, high_price, low_price, trig




if __name__ == "__main__":
        
    info = filterData(dataBaseSaveFile = "stockData.db")
    
    info.autoSaveIndicators()
    info.populateSummaryData()
    
    
    
    









