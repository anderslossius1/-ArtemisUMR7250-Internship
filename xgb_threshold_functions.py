"""
XGBOOST FUNCTIONS FOR COMPUTING FAR THRESHOLDS
CONTAINS:
    PHASE 1:
    - compute_chi_threshold() --> xgb_evaluation_functions.py
    - compute_p_lambda_threshold() --> xgb_evaluation_functions.py
    - compute_L_lambda_threshold() --> xgb_evaluation_functions.py

@author: alossius & mgorlach
"""
import numpy as np

def compute_chi_threshold(far_input, bkg_time, chi, y_test):
    """PHASE 1
    Determines the threshold to apply on the xgboost-generated prediction probabilites. 

    Parameters
    ----------
    far_input : float
        The desired false alarm rate to compute the threshold for.
    bkg_time : float
        A fraction of the total background lifetime generated from the PySTAMPAS bck postprocessing.
    chi : array
        Array containing prediction probabilities for each trigger in the testing dataset
    y_test : Pandas Series
        An array containing the true identities (signal or bkg) for each trigger in the testing dataset

    Returns
    -------
    p : float
        The xgboost probability threshold value. 
    far : float
        The adjusted false alarm rate that was actually used to determine the threshold. 

    """
    # Assess passed false alarm rate, compute new FAR if needed
    if far_input < (1/bkg_time):
        print('Invalid false alarm rate. Using minimum FAR possible: ', 1/bkg_time, 'Hz')
        far_toUse = 1/bkg_time
        
    elif ((far_input % (1/bkg_time)) / (1/bkg_time) > 1e-16):
        far1 = 0
        
        while(abs(far1-far_input) > (1/bkg_time)): far1 += (1/bkg_time) 
        
        far2 = far1 + (1/bkg_time) 
        
        if   (abs(far_input - far1) > abs(far_input - far2)): far_toUse = far2
        elif (abs(far_input - far1) < abs(far_input - far2)): far_toUse = far1
        else                                                : far_toUse = far2 #or far1
        
        print('False alarm rate being used =', far_toUse, 'Hz')
        
    else: far_toUse = far_input # Nothing wrong with inputed FAR
    
    # Compute threshold corresponding to false alarm rate
    min, max = 0, 1
    p = (min+max)/2
    
    far = sum(((chi>p)-y_test)==1)/bkg_time
    
    while (abs(far-far_toUse)/far_toUse > 0.001):
        
        if (far > far_toUse): 
            min = p
            p = (min+max)/2
        elif (far < far_toUse): 
            max = p
            p = (min+max)/2 
        else: return p
        
        far = sum(((chi>p)-y_test)==1)/bkg_time

    return p, far
    
def compute_p_lambda_threshold(far_input, bkg_time, df_test):
    """PHASE 1
    Determines the threshold to apply on the pystampas-generated detection statistic p_lambda. 

    Parameters
    ----------
    far_input : float
        The desired false alarm rate to compute the threshold for.
    bkg_time : float
        A fraction of the total background lifetime generated from the PySTAMPAS bck postprocessing.
    df_test : Pandas DataFrame
        DataFrame containing all triggers used for testing.

    Returns
    -------
    thr : float
        The pystampas p_lambda threshold value. 
    far : float
        The adjusted false alarm rate that was actually used to determine the threshold. 

    """
    # Assess passed false alarm rate, compute new FAR if needed
    if (far_input < (1/bkg_time)):
        print('Invalid false alarm rate. Using minimum FAR possible: ', 1/bkg_time, 'Hz')
        far_toUse = 1/bkg_time
        
    elif ((far_input % (1/bkg_time)) / (1/bkg_time) > 1e-16):
        far1 = 0
        
        while(abs(far1-far_input) > (1/bkg_time)): far1 += (1/bkg_time) 
        
        far2 = far1 + (1/bkg_time) 
        
        if   (abs(far_input - far1) > abs(far_input - far2)): far_toUse = far2
        elif (abs(far_input - far1) < abs(far_input - far2)): far_toUse = far1
        else                                                : far_toUse = far1 #or far2
        
        print('False alarm rate being used =', far_toUse, 'Hz')
        
    else: far_toUse = far_input # Nothing wrong with inputed FAR
    
    # Compute threshold corresponding to false alarm rate
    bkg_plambda = df_test[df_test['kind']=='bkg']['p_lambda']
    
    min = np.min(bkg_plambda)
    max = np.max(bkg_plambda)
    
    thr = (min+max)/2
    far = sum(bkg_plambda > thr) / bkg_time
    
    while (abs(far - far_toUse)/far_toUse > 0.001):
        
        if (far > far_toUse):
            min = thr
            thr = (min+max)/2
        elif (far < far_toUse):
            max = thr
            thr = (min+max)/2
        else: 
            return thr
        
        far = sum(bkg_plambda > thr) / bkg_time
    
    return thr, far

def compute_L_lambda_threshold(far_input, bkg_time, df_test):
    """PHASE 1
    Determines the threshold to apply on the pystampas-generated detection statistic L_lambda. 

    Parameters
    ----------
    far_input : float
        The desired false alarm rate to compute the threshold for.
    bkg_time : float
        A fraction of the total background lifetime generated from the PySTAMPAS bck postprocessing.
    df_test : Pandas DataFrame
        DataFrame containing all triggers used for testing.

    Returns
    -------
    thr : float
        The pystampas L_lambda threshold value. 
    far : float
        The adjusted false alarm rate that was actually used to determine the threshold. 

    """
    # Assess passed false alarm rate, compute new FAR if needed
    if (far_input < (1/bkg_time)):
        print('Invalid false alarm rate. Using minimum FAR possible: ', 1/bkg_time, 'Hz')
        far_toUse = 1/bkg_time
        
    elif ((far_input % (1/bkg_time)) / (1/bkg_time) > 1e-16):
        far1 = 0
        
        while(abs(far1-far_input) > (1/bkg_time)): far1 += (1/bkg_time)
        
        far2 = far1 + (1/bkg_time)
        
        if   (abs(far_input - far1) > abs(far_input - far2)): far_toUse = far2
        elif (abs(far_input - far1) < abs(far_input - far2)): far_toUse = far1
        else                                                : far_toUse = far1 #or far2 
        
        print('False alarm rate being used =', far_toUse, 'Hz')
        
    else: far_toUse = far_input # Nothing wrong with inputed FAR
    
    # Compute threshold corresponding to false alarm rate
    bkg_Llambda = df_test[df_test['kind']=='bkg']['L_lambda']
    
    min = np.min(bkg_Llambda)
    max = np.max(bkg_Llambda)
    
    thr = (min+max)/2
    far = sum(bkg_Llambda > thr) / bkg_time
    
    while (abs(far - far_toUse)/far_toUse > 0.001):
        
        if (far > far_toUse):
            min = thr
            thr = (min+max)/2
        elif (far < far_toUse):
            max = thr
            thr = (min+max)/2
        else: 
            return thr
        
        far = sum(bkg_Llambda > thr) / bkg_time
    
    return thr, far