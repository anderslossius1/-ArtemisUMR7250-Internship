"""
XGBOOST MODEL-EVALUATION FUNCTIONS
CONTAINS:
    PHASE 1:
    - evaluate() --> run_xgboost_training.py

@author: mgorlach & alossius
"""
import numpy as np
import pandas as pd
from pystampas.params import read_inj_params
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from xgb_efficiency_functions import compute_eff, compute_improvement_error, compute_hrss_50
from xgb_threshold_functions import compute_L_lambda_threshold, compute_p_lambda_threshold, compute_chi_threshold

def evaluate(df_test, xgb_cl, far, bkg_time, inj_p, waveforms):   
    """PHASE 1
    Classifier evaluation function. Quantifies the performance of the xgboost classifer on testing data.

    Parameters
    ----------
    df_test : Pandas DataFrame
        DataFrame containing all triggers to be used for testing. 
    xgb_cl : xgboost.XGBClassifer
        XGBoost classifer to be used for prediction and evaluation.
    far : float
        Desired false alarm rate for background analysis.
    bkg_time : float
        Fraction of the total background lifetime.
    inj_p : dict
        Injection params file
    waveforms : list
        List of the injected waveforms for the Phase 1 project

    Returns
    -------
    dict
        All created variables required for external use. 

    """
    # Split independent and dependent data
    X = df_test.drop(['Signal','alpha','kind'], axis=1).copy()  
    y = df_test['Signal'].copy() 
    
    # Run classifier prediction
    chi = xgb_cl.predict_proba(X)[:, 1] 

    # Compute the precision-recall machine learning metric
    average_precision = average_precision_score(y, chi)
    precision, recall, thresholds = precision_recall_curve(y, chi)
    auc_precision_recall = auc(recall, precision)

    # Compute FAR thresholds
    chi_threshold, far = compute_chi_threshold(far, bkg_time, chi, y)
    p_lambda_threshold = compute_p_lambda_threshold(far, bkg_time, df_test)[0]
    L_lambda_threshold = compute_L_lambda_threshold(far, bkg_time, df_test)[0]
    
    # Initialize arrays
    alpha       = [[]for _ in range(len(waveforms))] 
    dist        = [[]for _ in range(len(waveforms))] 
    hrss        = [[]for _ in range(len(waveforms))] 
    tot_inj     = [[]for _ in range(len(waveforms))] 
    eff_pys     = [[]for _ in range(len(waveforms))]
    eff_xgb     = [[]for _ in range(len(waveforms))] 
    eff_max     = [[]for _ in range(len(waveforms))] 
    eff_err_pys = [[]for _ in range(len(waveforms))]
    eff_err_xgb = [[]for _ in range(len(waveforms))] 
    eff_err_max = [[]for _ in range(len(waveforms))]
    truepos_pys = [[]for _ in range(len(waveforms))] 
    truepos_xgb = [[]for _ in range(len(waveforms))] 
    alpha50_xgb = [[]for _ in range(len(waveforms))]
    alpha50_pys = [[]for _ in range(len(waveforms))]
    alpha50_max = [[]for _ in range(len(waveforms))]
    dist50_xgb  = [[]for _ in range(len(waveforms))]
    dist50_pys  = [[]for _ in range(len(waveforms))]
    dist50_max  = [[]for _ in range(len(waveforms))]
    hrss_0      = []
    dist_0      = []
    mean_freqs  = []
    durations   = []

    # Use threshold to compute TP,FP,TN,FN for xgboost and pystampas
    xgb_predict = (chi >= chi_threshold) + 2 * y
    xgb_all_truepos = xgb_predict == 3
    xgb_all_trueneg = sum(xgb_predict == 0)
    xgb_all_falspos = sum(xgb_predict == 1)
    xgb_all_falsneg = sum(xgb_predict == 2)
    
    pys_predict = (df_test['p_lambda'] >= p_lambda_threshold) + 2 * y
    pys_all_truepos = pys_predict == 3
    pys_all_trueneg = sum(pys_predict == 0)
    pys_all_falspos = sum(pys_predict == 1)
    pys_all_falsneg = sum(pys_predict == 2)
    
    # Compute ML statisitics
    P = sum(y==1)
    N = sum(y==0)
    prevalence = P/(P+N)
    
    xgb_precision = sum(xgb_all_truepos) / (sum(xgb_all_truepos) + xgb_all_falspos)
    pys_precision = sum(pys_all_truepos) / (sum(pys_all_truepos) + pys_all_falspos)
    
    xgb_recall = sum(xgb_all_truepos) / (sum(xgb_all_truepos) + xgb_all_falsneg)
    pys_recall = sum(pys_all_truepos) / (sum(pys_all_truepos) + pys_all_falsneg)
    
    xgb_false_positive_rate = xgb_all_falspos / (xgb_all_falspos + xgb_all_trueneg)
    pys_false_positive_rate = pys_all_falspos / (pys_all_falspos + pys_all_trueneg)
    
    xgb_false_alarm_rate = xgb_all_falspos / bkg_time
    pys_false_alarm_rate = pys_all_falspos / bkg_time
    
    # Iterate through every waveform that was injected
    for i, wvf in enumerate(waveforms):
        
        # Obtain all the alpha values injected for waveform[i]
        alpha[i] = np.sort(df_test.loc[df_test['kind']==wvf]['alpha'].unique())
        
        # Compute distance and hrss property of waveform[i]
        inj_params = read_inj_params(wvf)
        d0 = inj_params['DISTANCE']
        hrss0 = inj_params['HRSS']
        
        hrss_0.append(format(hrss0, '.2e'))
        dist_0.append(format(d0, '.2f'))
        
        dist[i] = d0     /  np.sqrt(alpha[i])
        hrss[i] = hrss0  *  np.sqrt(alpha[i])
        
        # Iterate through each alpha factor of waveform[i]
        for j in range(len(alpha[i])):
            # Indexes of all triggers in df_test with alpha values equal to alpha[j] for waveform[i]
            indexes = df_test[df_test['alpha']==alpha[i][j]]  
            indexes = indexes[indexes['kind']==wvf].index    
            
            # Number of true positives registered for alpha[j] of waveform[i]
            truepos_pys[i] += [sum(pys_all_truepos[indexes])] 
            truepos_xgb[i] += [sum(xgb_all_truepos[indexes])] 
            
            # Number of injections originally injected for alpha[j] of waveform[i]
            tot_inj[i] += [len(indexes)] 
            
            # Compute efficiency and error on efficiency
            eff_p, eff_err_p = compute_eff(truepos_pys[i][j],  tot_inj[i][j])
            eff_x, eff_err_x = compute_eff(truepos_xgb[i][j],  tot_inj[i][j])
            eff_m, eff_err_m = compute_eff(sum(y[indexes]==1), tot_inj[i][j])
            
            eff_pys[i] += [eff_p]; eff_err_pys[i] += [eff_err_p]
            eff_xgb[i] += [eff_x]; eff_err_xgb[i] += [eff_err_x]
            eff_max[i] += [eff_m]; eff_err_max[i] += [eff_err_m]
        
            print('Progress ' + str(j+1) + '/' + str(len(alpha[i])) + ' for ' + str(i+1) + '/' + str(len(waveforms)) + ' ', end='\r')
    
    #THIS COULD BE IMPROVED#
    # Compute hrss@50%
    hrss_50_xgb = compute_hrss_50(eff_xgb, hrss)
    hrss_50_pys = compute_hrss_50(eff_pys, hrss)
    hrss_50_max = compute_hrss_50(eff_max, hrss)
    
    # Compute dist@50% and obtain mean freq and duration for each waveform
    for i, wvf in enumerate(waveforms):
        inj_params = read_inj_params(wvf)
        d0 = inj_params['DISTANCE']
        hrss0 = inj_params['HRSS']
        
        alpha50_xgb[i] = (hrss_50_xgb[i] / hrss0)**2
        alpha50_pys[i] = (hrss_50_pys[i] / hrss0)**2
        alpha50_max[i] = (hrss_50_max[i] / hrss0)**2
        dist50_xgb[i]  = d0 / np.sqrt(alpha50_xgb[i])
        dist50_pys[i]  = d0 / np.sqrt(alpha50_pys[i])
        dist50_max[i]  = d0 / np.sqrt(alpha50_max[i])
        
        mf = 0.5 * (inj_params['FMIN'] + inj_params['FMAX'])
        mean_freqs.append(mf)
        durations.append(inj_params['DURATION'])
        
    # Define the improvement over pystampas as the normalized difference between hrss@50% values
    xgb_improvement  = (hrss_50_pys - pd.DataFrame(hrss_50_xgb)[0]) / hrss_50_pys
    max_improvement  = (hrss_50_pys - pd.DataFrame(hrss_50_max)[0]) / hrss_50_pys
    
    # Compute improvement error
    err_improvement = compute_improvement_error(hrss, len(waveforms), truepos_xgb, tot_inj, eff_xgb, hrss_50_pys)
    
    # Print relative improvement over PySTAMPAS
    average_xgb_improvement = sum(xgb_improvement)    / len(xgb_improvement)
    average_max_improvement = sum(max_improvement)    / len(max_improvement)
    frac_of_maximum         = average_xgb_improvement / average_max_improvement
    print('\nTraining reached '+str(round(frac_of_maximum*100,1))+'% of the maximum possible (hrss_50) improvement.')

    # Define average efficiency of each waveform
    wvf_eff_pys = [sum(eff_pys[i])/len(eff_pys[i]) for i in range(len(eff_pys))]
    wvf_eff_xgb = [sum(eff_xgb[i])/len(eff_xgb[i]) for i in range(len(eff_xgb))]
    wvf_eff_improv = [{waveforms[i]:str(round((wvf_eff_xgb[i] - wvf_eff_pys[i])*100,2))+' %'} for i in range(len(wvf_eff_xgb))]

    # Return evaluation variables
    return {'X':X, 'y':y,
            'far':far,
            'hrss':hrss, 'dist':dist,
            'waveforms':pd.Series(waveforms), #gives order of waveforms in all data                                                                                                                                
            'wvf_eff_improv':wvf_eff_improv,
            'chi':chi,
            'ML_precision':precision, 'ML_recall':recall, 'ML_thresholds':thresholds,
            'frac_of_maximum':frac_of_maximum,
            'average_precision': float(average_precision), 'auc_precision_recall':float(auc_precision_recall),
            'prevalence':prevalence,
            'xgb_precision':xgb_precision, 'xgb_recall':xgb_recall, 'xgb_false_positive_rate':xgb_false_positive_rate, 'xgb_false_alarm_rate':xgb_false_alarm_rate,
            'pys_precision':pys_precision, 'pys_recall':pys_recall, 'pys_false_positive_rate':pys_false_positive_rate, 'pys_false_alarm_rate':pys_false_alarm_rate,
            'xgb_all_truepos':sum(xgb_all_truepos), 'pys_all_truepos':sum(pys_all_truepos),
            'xgb_all_trueneg':xgb_all_trueneg, 'xgb_all_falspos':xgb_all_falspos, 'xgb_all_falsneg':xgb_all_falsneg,
            'pys_all_trueneg':pys_all_trueneg, 'pys_all_falspos':pys_all_falspos, 'pys_all_falsneg':pys_all_falsneg,
            'eff_xgb':eff_xgb, 'eff_err_pys':eff_err_pys, 'hrss_50_xgb':np.array(hrss_50_xgb).tolist(), 'dist_50_xgb':np.array(dist50_xgb).tolist(), 'xgb_improvement':xgb_improvement, 'chi_threshold'    :chi_threshold,
            'eff_pys':eff_pys, 'eff_err_xgb':eff_err_xgb, 'hrss_50_pys':np.array(hrss_50_pys).tolist(), 'dist_50_pys':np.array(dist50_pys).tolist(), 'max_improvement':max_improvement, 'p_lambda_threshold':p_lambda_threshold,
            'eff_max':eff_max, 'eff_err_max':eff_err_max, 'hrss_50_max':np.array(hrss_50_max).tolist(), 'dist_50_max':np.array(dist50_max).tolist(), 'err_improvement':err_improvement, 'L_lambda_threshold':L_lambda_threshold,
            'dist0':dist_0, 'hrss0':hrss_0, 'mean_freqs':mean_freqs, 'durations':durations}