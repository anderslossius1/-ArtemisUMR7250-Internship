"""
XGBOOST FUNCTIONS TO QUANTIFY DETECTION EFFICIENCY
CONTAINS:
    PHASE 1:
    - compute_eff() --> xgb_evaluation_functions.py
    - compute_improvement_error --> xgb_evaluation_functions.py
    - compute_eff_error --> here
    - compute_hrss_50 --> xgb_evaluation_functions.py

@author: mgorlach & alossius
"""
import numpy as np
import pandas as pd
from math import factorial
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from xgb_utils import quad_curve, logistic_curve

def compute_eff(N_inj_recovered, N_inj_total):
    """PHASE 1
    Computes detection efficiency. 
 
    Parameters
    ----------
    N_inj_recovered : float
        Nunber of injections recovered by detection method.
    N_inj_total : float
        Total number of injections for current alpha factor.
 
    Returns
    -------
    array x2
        Arrays for both the detecion efficiencies and error on the efficiency. 
 
    """
    eff = N_inj_recovered / N_inj_total
    eff_err = compute_eff_error(N_inj_recovered, N_inj_total)
    
    return eff, eff_err

def compute_improvement_error(hrss, N_wvf, N_inj_recovered, N_inj_total, eff, hrss_50_pys):
    """PHASE 1
    Computes uncertainties on the percentage XGBoost improvement over PySTAMPAS (using hrss@50%)

    Parameters
    ----------
    hrss : array of float
        Array of each wvf's hrrs values.
    N_wvf : int
        Number of injected waveforms.
    N_inj_recovered : array of int
        Array of each wvf's XGBoost true-positives.
    N_inj_total : array of int
        Array of each wvf's number of injections.
    eff : array of float
        Array of each wvf's XGBoost efficiency.
    hrss_50_pys : array of float
        Array of each wvf's PySTAMPAS hrss@50% values.

    Returns
    -------
    err_improvement : array of float
        Array of each wvf's improvement error.

    """
    # Initialize
    uncert = [[] for _ in range(N_wvf)]
    minus  = []
    plus   = []
    
    # Iterating over all alpha values for each waveform
    for i in range(N_wvf):        
        
        for j in range(len(N_inj_total[i])): 
            uncert[i] += [compute_eff_error(N_inj_recovered[i][j], N_inj_total[i][j])]
        
        unc = np.array(uncert[i])
        
        popt, pcov  = curve_fit(quad_curve, np.array(eff[i])[unc!=0], unc[unc!=0], p0=[-0.5,1,0.022])
        unc[unc==0] = quad_curve(np.array(eff[i])[unc==0], popt[0], popt[1], popt[2])
        
        uncert[i] = unc.tolist()

        minus += [(np.array(eff[i]) - unc).tolist()]
        plus  += [(np.array(eff[i]) + unc).tolist()]
    
    # Compute all upper and lower hrss@50% values
    hrss_50_plus  = compute_hrss_50(plus,  hrss)
    hrss_50_minus = compute_hrss_50(minus, hrss)
    
    # Compute error 
    err_hrss_50   = (pd.DataFrame(hrss_50_minus)[0] - pd.DataFrame(hrss_50_plus)[0])/2
    
    # Compute improvement error
    err_improvement = (err_hrss_50/pd.DataFrame(hrss_50_pys)[0]) 
    
    return err_improvement

def compute_eff_error(N_inj_recovered, N_inj_total):
     """PHASE 1
     Compute standard deviation of the estimator for detection efficiency using a Bayesian approach.
     Source: @AdrianMacquet/pystampas/postprocessing/efficiency_estimation.py
     """
     eff = np.linspace(0, 1, 1000)
     
     def p(eff_range, n_rec, n_inj):
         try: 
             result = factorial(n_inj+1) // (factorial(n_rec) * factorial(n_inj - n_rec))*eff_range**n_rec * (1-eff_range)**(n_inj-n_rec)
         except:
             result = [0]*len(eff_range)
         return result

     p_of_eff = p(eff, N_inj_recovered, N_inj_total)
     
     s_peff = np.sort(p_of_eff)
     res = np.cumsum(s_peff) / np.sum(p_of_eff)
     
     result = len(res[res>1-0.685]) / len(res) 
     error = result / 2

     return error   

def compute_hrss_50(eff, hrss):  
    """PHASE 1
    Compute hrss@50%
    Source: @AdrianMacquet/pystampas/postprocessing/efficiency_estimation.py

    Parameters
    ----------
    eff : float 
        Array containing all waveform efficiencies. 
    hrss : float
        Array containing all waveform hrss. 

    Returns
    -------
    hrss_50 : float
        Array of hrss values at 50% detection efficiency. 

    Notes
    -----
    Two methods are used to modelize eff(hrss). The first one tries to fit a sigmoid to the
    data while the second one uses an interpolation.

    """
    hrss_50   = np.zeros(len(eff))
    hrss_50_2 = np.zeros(len(eff))
    
    for j in range(len(eff)):
        # Compute hrss@50% with fitted curve
        try:
            popt, pcov = curve_fit(logistic_curve, np.log10(hrss[j]), eff[j], p0=[5, np.mean(np.log10(hrss[j]))])
            hrss_50[j] = 10**popt[1]
        except:
            hrss_50[j] = -1
            print('Warning : fit for efficiency curve did not converge. Using interpolate instead.')
    
        # Compute hrss@50% with interpolation
        x = np.log10(hrss[j])
        f = interp1d(x, eff[j])
        xx = np.linspace(np.min(x), np.max(x), 1000)
        yy = np.abs(f(xx) - 0.5)
        i_min = np.argmin(yy)
        
        hrss_50_2[j] = 10**xx[i_min]
            
        if hrss_50[j] == -1:
            hrss_50[j] = hrss_50_2[j]
        else:
            if not np.isclose(np.log10(hrss_50[j]), np.log10(hrss_50_2[j]), atol=0.2):
                print('Warning : curve fit and interpolation give different values for hrss50 :')
                print('Curve fit : ' + format(hrss_50[j]), '.2e')
                print('Interpolation : ' + format(hrss_50_2[j]), '.2e')
                print('You should check efficiency plot.')
                # hrss_50[j] = hrss_50_2[j] uncomment if efficiency curves could be improved by using interpolation 

    return hrss_50
