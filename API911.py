import pandas as pd
import matplotlib.pyplot as plt
import plotly
import numpy as np
import seaborn as sns
import calendar
import pandasql as pdsql
from scipy.stats import linregress
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import  bootstrap_plot
pylab.rcParams['figure.figsize'] = (15, 10)

#import matplotlib.pyplot as plt
#import plotly
#import seaborn as sns
#import calendar
#from scipy.stats import linregress
#from pandas.tools.plotting import autocorrelation_plot
#from pandas.tools.plotting import lag_plot
#from pandas.tools.plotting import  bootstrap_plot
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import requests
import io
import os 
import time
import pickle


#pysql = lambda q: pdsql.sqldf(q, globals())
#pysql_loc = lambda q: pdsql.sqldf(q, locals())
def load_911db_realtime():    
    url="https://storage.googleapis.com/montco-stats/tz.csv"
    d=requests.get(url).content
    d=pd.read_csv(io.StringIO(d.decode('utf-8')), header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','e'],dtype={'lat':float,'lng':float,'desc':str,'zip':str,'title':str,'timeStamp':datetime.datetime,'twp':str,'e':int})   
    d=pd.DataFrame(d)
    return d

def save_911db(df, dir, name):
    df.to_csv(dir + name)
    
def load_db_local():
    d = [];
    d=pd.read_csv('../data/911_MONTGOMERY_CALLS_MEDIUM.csv', header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','e'],dtype={'lat':float,'lng':float,'desc':str,'zip':str,'title':str,'timeStamp':datetime.datetime,'twp':str,'e':int})    
    d=pd.DataFrame(d)
    return d
    
def add_data_time_columns(dt): 
    
    # ADD THE DATE AND TIME IN SINGLE COLUMNS FOR GROUPPING THE CALLS BY HOURS
    dt1 = np.zeros((dt.shape[0], 4))

    for i in range(0,dt.shape[0]):
        s = dt.timeStamp[i]
        s = s.split(' ')
        ymd = s[0]. split('-')
        hms = s[1]. split(':')
        dt1[i,0]=int(ymd[0])
        dt1[i,1]=int(ymd[1])
        dt1[i,2]=int(ymd[2])
        dt1[i,3]=int(hms[0])

    dt1 = pd.DataFrame(dt1)
    dt2 = pd.concat([dt,dt1], axis=1)

    names = dt2.columns.tolist()
    names[names.index(0)] = 'year'
    names[names.index(1)] = 'month'
    names[names.index(2)] = 'day'
    names[names.index(3)] = 'hour'
    dt2.columns = names
    dt2 = pd.DataFrame(dt2)
    
    # RE-INDEX WITH THE TIME STAMP
    #pysql_loc = lambda q: pdsql.sqldf(q, locals())
    #dt3 = pysql_loc("select timeStamp, year, month, day, hour, count(*) as calls from TAB1 where title like 'EMS:%' group by year, month, day, hour")
    sql1 = "select timeStamp, year, month, day, hour, count(*) as calls from dt2 where title like 'EMS:%' group by year, month, day, hour"
    dt3 = pdsql.sqldf(sql1, {'dt2':dt2})
    dt3 = pd.DataFrame(dt3)
    dt4 = dt3.set_index('timeStamp')
    del dt4.index.name
    
    return dt4

# ---------------
# LOAD DB
# --------------
df_all = load_911db_realtime()
df_model = add_data_time_columns(dt)