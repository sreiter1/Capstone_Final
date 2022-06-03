# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 08:14:11 2022

@author: sreit
"""


import datetime
import pandas as pd


from flask import Flask, jsonify, render_template, flash, request, redirect, url_for
from flask_wtf import FlaskForm
import model
from wtforms import StringField, SubmitField, RadioField
from wtforms.validators import InputRequired
import os



class conversionTables:
    tickerConversionTable =  {"alphatime"   : "alpha_daily",
                              "balance"      : "balance_sheet",
                              "cash"         : "cash_flow",
                              "earnings"     : "earnings",
                              "overview"     : "fundamental_overview",
                              "income"       : "income_statement"}
    
    
    # Converts user unput to the columns in the table.  Provides a filter to 
    # prevent database corruption.
    dailyConversionTable =  {"OPEN"         : "open",
                             "CLOSE"        : "close",
                             "HIGH"         : "high",
                             "LOW"          : "low",
                             "ADJCLOSE"     : "adj_close",
                             "VOLUME"       : "volume",
                             "DIVIDEND"     : "dividend",
                             "SPLIT"        : "split",
                             "ADJRATIO"     : "adjustment_ratio",
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
                             "RSI"          : "rsi",
                             "IDEAL"        : "ideal_return",
                             "IDEAL_HIGH"   : "ideal_high",
                             "IDEAL_LOW"    : "ideal_low",
                             "IDEAL_TRIG"   : "ideal_return_trig"}
    
    
    indicatorList = {"MA20"        : "mvng_avg_20", 
                     "MA50"        : "mvng_avg_50", 
                     "MACD12"      : "macd_12_26", 
                     "MACD19"      : "macd_19_39",
                     "OBV"         : "on_bal_vol", 
                     "RSI"         : "rsi",
                     "BOLLINGER20" : "bollinger_20",
                     "BOLLINGER50" : "bollinger_50",
                     "IDEAL"       : "ideal_return"}
    
    
    def loadStockListCSV(self, stockListFileName, saveToDB = True):
        # reads a csv file of stock tickers and optionally saves them to 
        # 'ticker_symbol_list' table in the database.
        try:
            # Open the csv-based list of tickers/companies
            stockFile = open(stockListFileName, "r") 
            
        except:
            print("Bad stock list file.  Unable to open.")
            return 1
        
        # read each line and create a list for the outputs
        Lines = stockFile.readlines()
        
        DF_ticker = [] # ticker symbol list
        DF_name = [] # name of the company
        DF_exchange = [] # exchange that the stock is traded on
        DF_recordDate = [] # date that the ticker was added to the database
        
        # open each line, split on the comma to get each value, append the 
        # ticker, name, and exchange from the CSV to the lists, and add today's
        # date to the record date.  
        for line in Lines:
            stock = line.split(",")
            DF_ticker.append(stock[0])
            DF_name.append(stock[1])
            DF_exchange.append(stock[2].strip('\n'))
            DF_recordDate.append(datetime.date.today())
            
            # execute a save to the 'ticker_symbol_list' table.
            if saveToDB:
                self._updateTickerList(ticker_symbol = stock[0],
                                       name = stock[1],
                                       exchange = stock[2].strip("\n"),
                                       recordDate = str(datetime.date.today()))
            
        # create the dataframe with all the recorded data
        df = pd.DataFrame([DF_ticker,
                           DF_recordDate,
                           DF_name,
                           DF_exchange])
        
        # label the data
        df.index = ["ticker_symbol", "recordDate", "name", "exchange"]
        df = df.transpose()
        
        # return the data
        return df
    



class validationFunctions:
    
    # confirms that the value is a boolean
    def validateBool(self, value):
        if not (isinstance(value, bool)):
            raise ValueError("Input must be of type 'bool'.")
        else:
            return 1
    
    # confirms that the ticker list is a python list of text strings
    def validateListString(self, inputList = []):
        if not isinstance(inputList, list):
            raise TypeError("Input not a list")
        elif inputList == []:
            raise ValueError("Input list is empty")
        if not all(isinstance(item, str) for item in inputList):
            raise TypeError("Input not a list of strings.\n")
        
        return 1
    
    
    def validateListInt(self, inputList = []):
        if not isinstance(inputList, list):
            raise TypeError("Input not a list")
        elif inputList == []:
            raise ValueError("Input list is empty")
        if not all(isinstance(item, int) for item in inputList):
            raise TypeError("Input not a list of integers.\n")
        
        return 1
    
    
    def validateListFloat(self, inputList = []):
        if not isinstance(inputList, list):
            raise TypeError("Input not a list")
        elif inputList == []:
            raise ValueError("Input list is empty")
        if not all((isinstance(item, float) or isinstance(item, int)) for item in inputList):
            raise TypeError("Input not a list of numeric.\n")
        
        return 1
    
    
    def validateDateString(self, date_text):
        # Checks the string entered to see if it can be parsed into a date.
        try:
            datetime.datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYY-MM-DD")
            
        return 1
    
    
    def validateString(self, string):
        if not isinstance(string, str):
            raise TypeError("Input not a string.\n")
        else:
            return 1
        
        
    def validateInteger(self, integer):
        if not isinstance(integer, int):
            raise TypeError("Input is not an integer.\n")
        else:
            return 1
        
        
    def validateNum(self, number):
        if not (isinstance(number, float) or isinstance(number, int)):
            raise TypeError("Input is not an integer.\n")
        else:
            return 1
            
            



class callLimitExceeded(Exception):
    # raised when the program detects that the call limit was exceeded, 
    # either because the limit set in 'getData' is exceeded or if the API 
    # returns a specific error.
    pass





        
    
    
    
    
    
    