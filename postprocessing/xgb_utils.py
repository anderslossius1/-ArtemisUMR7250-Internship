"""
XGBOOST UTILITY AND MSC FUNCTIONS
CONTAINS:
    PHASE 1:
    - convertFreq() --> xgb_preproc_functions.py
    - logistic_curve() --> xgb_efficiency_functions.py, xgb_plot_functions.py
    - quad_curve() --> xgb_efficiency_functions.py
    OTHER:
    - timestart() and timestop() --> unused, can be used to time any aspect of any code
    - Spinner() Class --> run_xgboost_training.py
    
@author: mgorlach & alossius
"""
import sys
import time
import threading
import numpy as np
import pandas as pd
from timeit import default_timer

def convertFreq(df):
    """Converts a DataFrame with Fmin and Fmax columns to have Fmean and Frange columns instead/
    
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame with minimum and maximum frequency data.
        
    Returns
    -------
    df : Pandas DataFrame
        DataFrame with mean and range frequency data. 
    """
    df = pd.concat([df,
                    pd.DataFrame(
                        {'fmean' : (df['fmin']+df['fmax'])/2,
                         'frange' : df['fmax']-df['fmin']})],
                   axis=1)
    df.drop(['fmin','fmax'],axis=1,inplace=True)
    
    return df             

def logistic_curve(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))   

def quad_curve(x, a, b, c):
    return a*x**2+b*x+c 

def timestart():
    global timer_start
    timer_start = default_timer()

def timestop(string):
    timer_end = default_timer()
    time1 = timer_end-timer_start
    print(string, " : ", time1,"s\n")

class Spinner: #SOURCED EXTERNALLY - NOT SURE IF THAT MATTERS
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1: 
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False