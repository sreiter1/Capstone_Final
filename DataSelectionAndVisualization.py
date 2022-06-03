# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:16:33 2021

@author: sreit
"""


import pandas as pd
import datetime
import sqlite3
import warnings




class badInput(Exception):
    # Raised when the input value is too small
    pass




class filterData:
    def __init__(self, dataBaseSaveFile = "./SQLiteDB/stockData.db"):
        self.DB = sqlite3.connect(dataBaseSaveFile)
        self._cur = self.DB.cursor()
        self._tickerList = []   # Empty list that gets filled with a list of tickers to be considered
        
        # Converts a user string to the names of tables, and for 'ticker_symbol_list'
        self._tickerConversionTable = {"time"     : "daily_adjusted",
                                       "balance"  : "balance_sheet",
                                       "cash"     : "cash_flow",
                                       "earnings" : "earnings",
                                       "overview" : "fundamental_overview",
                                       "income"   : "income_statement"}
        
        # Converts user unput to the columns in the table.  Provides a filter to 
        # prevent database corruption.
        self._dailyConversionTable = {"OPEN"         : "open",
                                      "CLOSE"        : "close",
                                      "HIGH"         : "high",
                                      "LOW"          : "low",
                                      "ADJCLOSE"     : "adj_close",
                                      "VOLUME"       : "volume",
                                      "SPLIT"        : "split",
                                      "ADJRATIO"     : "adjustment_ratio",
                                      "MA20"         : "mvng_avg_20",
                                      "MA50"         : "mvng_avg_50",
                                      "MACD12"       : "macd_12_26",
                                      "MACD19"       : "macd_19_39",
                                      "VOL20"        : "vol_avg_20",
                                      "VOL50"        : "vol_avg_50",
                                      "OBV"          : "on_bal_vol",
                                      "DAYCHANGE"    : "percent_cng_day",
                                      "TOTALCHANGE"  : "percent_cng_tot",
                                      "RSI"          : "rsi"}
        
        
    
    def checkTickersInDatabase(self, 
                              returnError = False, 
                              returnDate = False,
                              returnNumRecord= False, 
                              tickerList = []):
        
        # This function confirms that tickers that are in the self._tickerList
        # are also in the 'ticker_symbol_list'.  Can optionally return the other 
        # data contained in the table as a pandas dataframe.
        
        # check function inputs for problematic values
        if not (isinstance(returnError, bool) or \
                isinstance(returnDate, bool) or \
                isinstance(returnNumRecord, bool)):
            raise ValueError("'return' attributes must be of type 'bool'.")
        
        
        if not isinstance(tickerList, list):
            raise TypeError("tickerlist not a list.")
        elif tickerList == []:
            raise ValueError("Internal list of tickers is empty.  Please run 'filterStocksFromDataBase()'")
        if not all(isinstance(item, str) for item in tickerList):
            raise TypeError("tickerlist not a list of strings.\n")
        
        
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
                returnWhat.append("error_daily_adjusted")
                returnWhat.append("error_balance_sheet")
                returnWhat.append("error_cash_flow")
            if returnDate:
                returnWhat.append("date_fundamental_overview")
                returnWhat.append("date_income_statement")
                returnWhat.append("date_earnings")
                returnWhat.append("date_daily_adjusted")
                returnWhat.append("date_balance_sheet")
                returnWhat.append("date_cash_flow")
            if returnNumRecord:
                returnWhat.append("records_fundamental_overview")
                returnWhat.append("records_income_statement")
                returnWhat.append("records_earnings")
                returnWhat.append("records_daily_adjusted")
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
            results = results.append(pd.DataFrame.from_records(data = query.fetchall(), columns = cols))
        
        # make the list of tickers in the dataframe available outside the funciton
        self._tickerList = results["ticker_symbol"].tolist()
        
        # return the dataframe of data
        return results
        
        
    
    def autoSaveIndicators(self,
                           indicators = None,
                           priceData = "ADJCLOSE",
                           useAdjVolume = False, 
                           tickerList = []):
        
        # Function calculates the indicators and saves them to the database.
        # Works on tickers contained in self._tickerList, for the indicators
        # passed to the function, and on the price passed to the function.  
        # Defaults to the adjusted close price and all the indicators.
        
        if isinstance(indicators, str):
            indicators = ["".join(e for e in indicators if e.isalpha())] # verify that all the text is alphabetical
        if indicators == None:
            indicators = ["ADJRATIO", "MA20", "MA50", "MACD12", "MACD19", \
                          "VOL20", "VOL50", "OBV", "DAYCHANGE", "TOTALCHANGE", "RSI"]
        
        if not isinstance(tickerList, list):
            raise TypeError("tickerlist not a list.")
        elif tickerList == []:
            tickerList = self._tickerList
            if tickerList == []:
                raise ValueError("Internal list of tickers is empty.  Please run 'filterStocksFromDataBase()'")
        if not all(isinstance(item, str) for item in tickerList):
            raise TypeError("tickerlist not a list of strings.\n")
                
                
        # Verify the tickers that are being searched are actually in the database.
        self.checkTickersInDatabase(tickerList = tickerList)
        
        
        # create the SQL transaction string, starting with the data that needs to
        # be extracted from 'daily_adjusted'
        columnString = ""
        if ("OBV" in indicators) or ("AdjustRatio" in indicators):
            columnString += ", close, adj_close, volume"
            
        if priceData not in ["CLOSE", "ADJCLOSE", "VOLUME"]:
            columnString += ", " + self._dailyConversionTable[priceData]
        
        # Request the data for each ticker in turn
        data = pd.DataFrame()
        stockCount = 0
        for ticker in tickerList:
            stockCount += 1
            print("\rAdding daily indicators for  " + str(ticker).rjust(7) + "      (" + str(stockCount).rjust(6) + " of " + str(len(tickerList)).ljust(6) + ")         ", end = "")
            
            queryString  = "SELECT recordDate" + columnString
            queryString += " FROM daily_adjusted \n"
            queryString += "WHERE ticker_symbol = '" + ticker + "';\n"
            
            query = self._cur.execute(queryString)
            cols = [column[0] for column in query.description]
            results = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
            
            # convert the recordDate from a string to a datetime object, and set as the index
            results["recordDate"] = pd.to_datetime(results["recordDate"])
            results = results.set_index("recordDate")
            results.sort_index()
            
            # Copy data from results to lists
            priceHist = list(results[self._dailyConversionTable[priceData]])
            closeHist = list(results[self._dailyConversionTable["CLOSE"]])
            adjCloseHist = list(results[self._dailyConversionTable["ADJCLOSE"]])
            volHist = list(results[self._dailyConversionTable["VOLUME"]])
            
            if useAdjVolume:
                volHist = [v * c / a for v, c, a in zip(volHist, closeHist, adjCloseHist)]
            
            # Calculate each indicator in turn and add the output to a dataframe
            data = pd.DataFrame()
                
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
                
                data[self._dailyConversionTable[indicator]] = series.values
            
            data = data.set_index(results.index)
            self._saveIndicatorsToDB(ticker_symbol = ticker,
                                     indicators = indicators,
                                     dataframe = data)
                
        return data
    
    
    
    def _saveIndicatorsToDB(self, ticker_symbol, indicators, dataframe):
        # Saves the data extracted from the price history (i.e. moving averages)
        # back to the database.  
        
        # start SQL query line
        sqlString  = "UPDATE daily_adjusted \n SET "
        sqlArray = pd.DataFrame()
        
        # add each indicator in the list of indicators to the SQL query, along
        # with the associated values.
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
    
    
    
    def populateFilterData(self):
        # the database has a table called "filter_cache" that contains some 
        # data from several tables in one location to make certain queries 
        # easier and faster.  This function ensures that all the tickers in 
        # "ticker_symbol_list" are contained in the "filter_cache"
        
        # get all the tickers in the database and compile them into a list
        query  = "SELECT ticker_symbol FROM ticker_symbol_list;\n"
        result = self._cur.execute(query)
        data = result.fetchall()
        tickerList = [item for t in data for item in t]
        
        # count the number of tickers for a command line update; set to 0 initially.
        populationCount = 0
        
        # for each ticker, increase the count by one, add the ticker to the filter_cache,
        # and make the calculated date blank
        for ticker in tickerList:
            populationCount += 1
            print("\rAdding " + str(ticker).rjust(7) + "      (" + str(populationCount).rjust(6) + " of " + str(len(tickerList)).ljust(6) + ")         ", end = "")
            query  = "INSERT OR IGNORE INTO filter_cache (ticker_symbol, date_calculated)\n"
            query += "VALUES(?, ?)"
            argList = [ticker, ""]
            self._cur.execute(query, argList)
        
        # commit the changes to the database
        self.DB.commit()
        
        # call the function that fills the "filter_cache" table in the database
        self._calculateFilterData()
        
        
        
    def _calculateFilterData(self, lastCalculated = -1):
        # the database has a table called "filter_cache" that contains some 
        # data from several tables in one location to make certain queries 
        # easier and faster.  This function ensures that all the tickers in 
        # "ticker_symbol_list" are contained in the "filter_cache"
        
        # get today's date as a string
        dateCalculated = str(datetime.datetime.now().date())
        
        # check function inputs for error / unexpected data.  Also sets 
        # "lastCalculated" based on the user input to be the date where 
        # anything with a refresh date older than this will get refreshed.  
        # Other records will be ignored.
        if isinstance(lastCalculated, str):
            self.validateDateString(lastCalculated)
            lastCalculated = datetime.datetime.strptime(lastCalculated, '%Y-%m-%d').date()
        elif isinstance(lastCalculated, int):
            lastCalculated = datetime.date.today()-datetime.timedelta(days = lastCalculated)
        
        # get a list of all the tickers in the "filter_cache" table from the database
        query  = "SELECT ticker_symbol, date_calculated FROM filter_cache;\n"
        result = self._cur.execute(query)
        data = result.fetchall()
        
        # convert the response to a list of lists, formatted such that each record 
        # is an entry in the outer list, while the inner list contains the ticker
        # and the calculated date
        filterData = [list(t) for t in data]  
        
        # counter and total variables for a command line progress update
        length_FilterDate = len(filterData)
        count_records = 0
        
        # 
        for record in filterData:
            # progress update
            count_records += 1
            print("\rWorking " + str(record[0]).rjust(7) + "    (" + str(count_records).rjust(6) + " of " + str(length_FilterDate).ljust(6) + ")             ", end = "")
            
            # the date is saved as a string in the database, so it needs to be 
            # converted back to a date-time object.  This function sets that 
            # datetime object as the date from the database or 01 Jan 1900.
            # This object is assigned to "calcDate"
            if (record[1] != ""):
                calcDate = datetime.datetime.strptime(record[1], '%Y-%m-%d').date()
            else:
                calcDate = datetime.datetime(1900, 1, 1).date()
            
            # check to see if the last time the data was refreshed is older than
            # the user input date, and if it is older, make updates to "filter_cache"
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
                queryString  = "SELECT recordDate, adj_close FROM daily_adjusted \n"
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
                    
                
                
                # Create a SQL update command to assign the values collected above
                # and assign them to specific columns in "filter_cache".  
                queryString  = "UPDATE filter_cache \n"
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
                queryString += "    pe_ratio = ?"
                queryString += "WHERE ticker_symbol = ?"
                
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
                           record[0]]
                
                
                # Execute and commit the SQL update
                self._cur.execute(queryString, argList)
                self.DB.commit()
    
    
    
    def validateDateString(self, date_text):
        # Checks the string entered to see if it can be parsed into a date.
        try:
            datetime.datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")
    
    
    
    def filterStocksFromDataBase(self, 
                                 updateBeforeFilter = False, # True ==> recompile data before executing filter; False ==> run on available
                                 lastCalculated = -1,         # how many days ago calculated filter data can be considered valid
                                 dailyLength = 0,            # time = 1250 is 5 years of trading days.  Default is keep all.
                                 marketCap = 0,              # commonStockSharesOutstanding from balance * price from daily
                                 sector = None,              # fundamental overview
                                 exchange = None,            # fundamental overview
                                 country = None,             # fundamental overview
                                 industry = None,            # fundamental overview
                                 peRatio = 0.0,              # EPS under earnings, price under daily
                                 profitMargin = 0.0,         # profit/loss in cash flow; net income / total revenue --> income statement
                                 shareHolderEquity = 0,      # balance sheet
                                 EPS = 0.0 ):                # from earnings
                       
        # Creates a list of stocks that meet the requirements passed to the function.
        
        # Check date in the filter table to ensure that it meets the "lastCalculated"
        # condition taken as an input.
        if updateBeforeFilter == True:
            self._calculateFilterData(lastCalculated = lastCalculated)
        
        # check inputs
        if not isinstance(dailyLength, int):
            raise TypeError("minHistory should be an integer.\n")
        if not isinstance(marketCap, int):
            raise TypeError("marketCap should be an integer.\n")
        if not isinstance(peRatio, float):
            raise TypeError("peRatio should be a float.\n")
        if not isinstance(profitMargin, float):
            raise TypeError("profitMargin should be a float.\n")
        if not isinstance(shareHolderEquity, int):
            raise TypeError("shareHolderEquity should be an integer.\n")
        if not isinstance(EPS, float):
            raise TypeError("EPS should be a float.\n")
        
        
        if sector is None:
            pass
        elif isinstance(sector, list): 
            if not all(isinstance(item, str) for item in sector):
                raise TypeError("sector not a list of ticker strings.\n")
        else:
            raise TypeError("sector not recognized list or None.\n")
            
        if exchange is None:
            pass
        elif isinstance(exchange, list): 
            if not all(isinstance(item, str) for item in exchange):
                raise TypeError("exchange not a list of ticker strings.\n")
        else:
            raise TypeError("exchange not recognized list or None.\n")
        
        if country is None:
            pass
        elif isinstance(country, list): 
            if not all(isinstance(item, str) for item in country):
                raise TypeError("country not a list of ticker strings.\n")
        else:
            raise TypeError("country not recognized list or None.\n")
            
        if industry is None:
            pass
        elif isinstance(industry, list): 
            if not all(isinstance(item, str) for item in industry):
                raise TypeError("industry not a list of ticker strings.\n")
        else:
            raise TypeError("industry not recognized list or None.\n")

        
        
        # start SQL query string.  "WHERE 1=1 AND ..." allows for the addition 
        # of other requirements within the request string later
        query  = "SELECT [COLUMNS]"
        query += "FROM filter_cache \n"
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
            warnings.warn("tickerList is an empty list.  Is the table 'filter_cache' empty?  Run filter fuction with option updateBeforeFilter = True")
        
        # return the dataframe
        return DF_tickerList
    
    
    
    def listUnique(self, extended = False):
        # returns a dictionary of the unique values available in the filter_cache table
        
        # start SQL query string.  "WHERE 1=1 AND ..." allows for the addition 
        # of other requirements within the request string later
        query  = "SELECT [COLUMNS]"
        query += "FROM filter_cache \n"
                
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
    
        

    def _macdExp(self, 
                 hist, 
                 period_1 = 12, 
                 period_2 = 26, 
                 periodSignal = 9):
        # calculate the moving average convergence-divergence using an exponential moving average
        
        if not isinstance(hist, list):
            raise ValueError("'hist' must be a list of historical pricing data.")
        if not isinstance(period_1, int) or period_1 <= 2:
            raise ValueError("'period_1' must be an integer of at least 2.")
        if not isinstance(period_2, int) or period_2 <= 2:
            raise ValueError("'period_2' must be an integer of at least 2.")
        if not isinstance(periodSignal, int) or periodSignal <= 2:
            raise ValueError("'periodSignal' must be an integer of at least 2.")
        if period_1 == period_2:
            raise ValueError("Periods should not be identical for MACD calculation")
            
        # swap the MACD period if the first is larger than the second
        elif period_1 < period_2:
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
        
        if not isinstance(hist, list):
            raise ValueError("'priceHist' must be a list of historical pricing data.")
        if not isinstance(period_1, int) or period_1 <= 2:
            raise ValueError("'period_1' must be an integer of at least 2.")
        if not isinstance(period_2, int) or period_2 <= 2:
            raise ValueError("'period_2' must be an integer of at least 2.")
        if not isinstance(periodSignal, int) or periodSignal <= 2:
            raise ValueError("'periodSignal' must be an integer of at least 2.")
        if period_1 == period_2:
            raise ValueError("Periods should not be identical for MACD calculation")
        
        # swap the MACD period if the first is larger than the second
        elif period_1 > period_2:
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
        
        if not isinstance(hist, list):
            raise ValueError("'priceHist' must be a list of historical pricing data.")
        if not isinstance(periods, int) or periods <= 2:
            raise ValueError("'periods' must be an integer of at least 2.")
        
        # initiate the geometric scaling factor for the exponential moving average
        # and set the initial moving average to be the first value in the price 
        # history
        multiplier = smoothing / (periods + 1)
        ema = [hist[0]]
        
        # for each value after the first, calculate the expoential moving average 
        # and append it to the list of moving average values
        for val in hist[1:]:
            nextEma = (ema[-1] * (1 - multiplier) +  val * multiplier)
            ema.append(nextEma)
        
        # return the list of the moving average values
        return ema



    def _simpleMovingAverage(self, 
                             hist, 
                             periods = 20):
        # calculate the simple moving average
        
        if not isinstance(hist, list):
            raise ValueError("'priceHist' must be a list of historical pricing data.")
        if not isinstance(periods, int) or periods <= 2:
            raise ValueError("'periods' must be an integer of at least 2.")
        
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



    def _rsi(self, 
             hist, 
             periods = 14,
             exp = False):
        # calculate the Relative Strength Indicator (RSI); not matching other dataset at barchart
        
        
        # check inputs for errors or problems
        if not isinstance(hist, list):
            raise ValueError("'priceHist' must be a list of historical pricing data.")
        if not isinstance(periods, int) or periods <= 2:
            raise ValueError("'periods' must be an integer of at least 2.")
            
        if len(hist) < periods+1:
            raise ValueError("Length of the history provided to RSI insufficient to meet period requirement.")
        
        
        # Calculate the percent change from one day to the next, and inert a '0' as 
        # the first value. 
        percentChangeArray = [(b-a)/a for a,b in zip(hist[:-1], hist[1:])]
        percentChangeArray.insert(0,0)
        
        # create a list of the gains (percent change > 0), and losses (percent 
        # change < 0).  The lists will correspond element-for-element with the
        # percent change array, and will have 0's to pad those elements that 
        # should not be in that list.
        gainArray = [ a if a > 0 else 0 for a in percentChangeArray]
        lossArray = [-a if a < 0 else 0 for a in percentChangeArray]
        
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
                incrementalLoss = ((incrementalLoss * (periods-1)) + lossArray[i]) / periods
                try:
                    rs = incrementalGain / incrementalLoss
                except:
                    rs = 0
                
                # calcuate rsi from the rs value and append it to the array
                rsi = 100 - (100 / (1 + rs))
                rsiArray.append(rsi)
            
        
        else:
            # For each element in the price history from "periods" (where the RSI 
            # becomes valid) through the end of the price history, calculate the
            # simple moving average of the gain and of the loss, calculate 
            # the RSI based on those values, and set RS = 0 if the incremental 
            # loss would cause a divide-by-zero error
            for i in range(periods, len(hist)):
                incrementalGain = sum(gainArray[i-periods:i]) / periods
                incrementalLoss = sum(lossArray[i-periods:i]) / periods
                try:
                    rs = incrementalGain / incrementalLoss
                except:
                    rs = 0
                
                # calcuate rsi from the rs value and append it to the array
                rsi = 100 - (100 / (1 + rs))
                rsiArray.append(rsi)
        
        # return the array of RSI values
        return rsiArray



    def _percentChangeDaily(self, hist):
        # calculate the percent change between days
        
        # check for unexpected inputs
        if not isinstance(hist, list):
            raise ValueError("'priceHist' must be a list of historical pricing data.")
        
        # calculate the percent change as (next - prev) / prev, and then insert
        # 0 as the first value
        percentChange = [(b-a)/a for a,b in zip(hist[:-1], hist[1:])]
        percentChange.insert(0, 0)
        
        return percentChange
    
    
    
    def _percentChangeTotal(self, hist):
        # calculate the percent change from the start of the data
        
        # check for unexpected inputs
        if not isinstance(hist, list):
            raise ValueError("'priceHist' must be a list of historical pricing data.")
        
        # calculate the percent change as (next - prev) / prev, and then insert
        # 0 as the first value
        percentChange = [(a-hist[0])/hist[0] for a in hist[1:]]
        percentChange.insert(0, 0)
        
        return percentChange

        

    def _stockAdjustRatio(self, 
                          close, 
                          adjClose):
        # calculate the total adjustment ratio between the close and adjusted close
        # prices.  
        
        if not isinstance(close, list):
            raise ValueError("'closePrice' must be a list of historical pricing data.")
        if not isinstance(adjClose, list):
            raise ValueError("'adjClose' must be a list of historical pricing data.")
        if len(close) != len(adjClose):
            raise ValueError("'closePrice' and 'adjClose' must have the same length.  ")
            
        adjustRatio = [c / a for c, a in zip(close, adjClose)]
        return adjustRatio
    
    
    
    def _onBalanceVolume(self, 
                         close, 
                         volume):
        # calculates the on-balance volume indicator
        
        if not isinstance(close, list):
            raise ValueError("'closePrice' must be a list of historical pricing data.")
        if not isinstance(volume, list):
            raise ValueError("'volume' must be a list of historical pricing data.")
        if not len(close) == len(volume):
            raise ValueError("'closePrice' and 'volume' must have the same length.")
        
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






if __name__ == "__main__":
        
    info = filterData(dataBaseSaveFile = "stockData.db")
    # info.populateFilterData()
    filteredList = info.filterStocksFromDataBase(dailyLength = 30)
    
        
    # df = info.filterStocksFromDataBase()
    data = info.autoSaveIndicators()









