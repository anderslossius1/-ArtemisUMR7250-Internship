"""
XGBOOST PLOTTING FUNCTIONS
CONTAINS:
    PHASE 1:
    - plot_fitted_curves() --> run_xgboost_training.py
    - plot_improvement_bar() --> run_xgboost_training.py
    - plotThresholds_p_lambda() --> run_xgboost_training.py
    - plotThresholds_L_lambda() --> run_xgboost_training.py
    - plot_conf_matrix() --> run_xgboost_training.py
    - plot_PR_curve() --> run_xgboost_training.py
    - plot_DetectionStatComparison() --> run_xgboost_training.py
    - save_a_tree() --> run_xgboost_training.py

@author: mgorlach & alossius
"""
import os
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from xgb_utils import logistic_curve
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import plot_precision_recall_curve, plot_confusion_matrix

def plot_fitted_curves(evaluated, plt_type, save_dir):
    """PHASE 1
    Plots the efficiency curves (max, xgb, pystampas) with their fits into one plot for each waveform.

    Parameters
    ----------
    evaluated : dict
        Dictionary including all information generated/computed from the evaluation.
    plt_type : str
        The independent variable to be used in the plots. Options are 'dist' or 'hrss'.
    save_dir : str
        Directory to save plots.

    Returns
    -------
    None.

    """
    print('\nPlotting efficiency curves...')
    
    # Create directory for efficiency curve plots
    os.makedirs(save_dir + '/efficiency_curves', exist_ok=True)
    
    # Determine independent variable
    if plt_type == 'hrss':
        x_axis    = evaluated['hrss']
        xlabel   = r'hrss $[1/\sqrt{Hz}]$'; ltype = 'hrss'
        x50_xgb, x50_pys, x50_max = evaluated['hrss_50_xgb'], evaluated['hrss_50_pys'], evaluated['hrss_50_max']
        save_type = '_hrss.png'
    if plt_type == 'dist':
        x_axis    = evaluated['dist']
        xlabel   = 'Distance [Mpc]'; ltype = 'dist'
        x50_xgb, x50_pys, x50_max = evaluated['dist_50_xgb'], evaluated['dist_50_pys'], evaluated['dist_50_max']
        save_type = '_dist.png'
    
    # Plot for all waveforms
    for j, wvf in enumerate(evaluated['waveforms']):

        print('Progress ' + str(j) + '/' + str(len(evaluated['waveforms'])) + ' ', end='\r')        

        x = x_axis[j]
        eff_xgb, eff_pys, eff_max = evaluated['eff_xgb'][j], evaluated['eff_pys'][j], evaluated['eff_max'][j]
        eff_err_xgb, eff_err_pys, eff_err_max = evaluated['eff_err_xgb'][j], evaluated['eff_err_pys'][j], evaluated['eff_err_max'][j]
        
        p0 = [5, np.mean(np.log10(x))]
        
        try:
            popt_xgb, pcov_xgb = curve_fit(logistic_curve,  np.log10(x),  eff_xgb,  p0=p0)
            popt_pys, pcov_pys = curve_fit(logistic_curve,  np.log10(x),  eff_pys,  p0=p0)
            popt_max, pcov_max = curve_fit(logistic_curve,  np.log10(x),  eff_max,  p0=p0)
        except:
            popt_xgb, popt_pys, popt_max = p0, p0, p0
        
        xx = np.linspace(np.min(np.log10(x)), np.max(np.log10(x)), 100)
        xxx = 10**xx    
        
        y_xgb = logistic_curve(xx, popt_xgb[0], popt_xgb[1])
        y_pys = logistic_curve(xx, popt_pys[0], popt_pys[1])
        y_max = logistic_curve(xx, popt_max[0], popt_max[1])
        
        plt.figure(figsize=(16,9))
        ax = plt.gca()
        ax.errorbar(x, eff_xgb, eff_err_xgb, marker='.', linestyle='', color='b', capsize=2)
        ax.errorbar(x, eff_pys, eff_err_pys, marker='.', linestyle='', color='g', capsize=2)
        ax.errorbar(x, eff_max, eff_err_max, marker='.', linestyle='', color='r', capsize=2)
        ax.plot(xxx, y_xgb, 'b')
        ax.plot(xxx, y_pys, 'g')
        ax.plot(xxx, y_max, 'r', linestyle='--')
        ax.vlines(x50_xgb[j], 0, 1, label='XGBoost   : ' + ltype + '@50% = {:.3g}'.format(x50_xgb[j]), color='b')
        ax.vlines(x50_pys[j], 0, 1, label='PySTAMPAS : ' + ltype + '@50% = {:.3g}'.format(x50_pys[j]), color='g')
        ax.vlines(x50_max[j], 0, 1, label='Maximum   : ' + ltype + '@50% = {:.3g}'.format(x50_max[j]), color='r', linestyle='--')
        ax.set_title('Fitted Efficiency Curve for ' + wvf)
        ax.set_xscale('log')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(r'Efficiency', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(which='both')
        ax.set_ylim(auto=True)
        
        plt.legend()
        plt.savefig(save_dir + '/efficiency_curves/' + wvf + save_type, dpi=400)
        plt.close()
        
    print('Efficiency curves saved to ' + save_dir + '/efficiency_curves/')
    
def plot_improvement_bar(evaluated, train_list, save_dir):
    """PHASE 1
    Creates bar chart of the improvement of all waveforms.

    Parameters
    ----------
    evaluated : dict
        Dictionary including all information generated/computed from the evaluation.
    train_list : list
        List with all waveforms used for training. 
    save_dir : str
        Directory to save plots.

    Returns
    -------
    None.

    """
    print('\nPlotting improvement bar chart...')
    
    # Obtain number of waveforms
    len_wvf = len(evaluated['waveforms'])
    
    # Plot horizontal bar chart
    plt.figure(figsize=(6, 8))
    plt.barh(range(len_wvf), evaluated['max_improvement']*100, color='white', edgecolor='k')
    plt.barh(range(len_wvf), evaluated['xgb_improvement']*100, xerr=evaluated['err_improvement']*100, color='C0', edgecolor='C0')
    plt.barh(range(len_wvf), evaluated['waveforms'].isin(train_list).astype(int)*evaluated['xgb_improvement']*100, color='C1', edgecolor='C1')
    plt.legend(['Maximum Possible','Not Trained On','Trained On'])
    plt.yticks(range(len_wvf), evaluated['waveforms'])
    plt.title('Improvement Against PySTAMPAS, FAR='+str(round(evaluated['far']*3600*24*365,2))+' per year') 
    plt.xlabel('Improvement in %')
    plt.tight_layout()
    plt.savefig(save_dir + '/improvement_bar.png', dpi=1000)
  
    print('Improvement bar chart saved to ' + save_dir + '/improvement_bar.png')
    
def plotThresholds_p_lambda(df_test, evaluated, bkg_time, save_dir):
    """PHASE 1
    Creates xgboost and pystampas trigger distribution for xgboost probability and pystampas p_lambda
    detection statiscs, along with visual representation of thresholds set for each based off the false alarm rate chosen. 

    Parameters
    ----------
    df_test : Pandas DataFrame
        DataFrame with all data used for testing. 
    evaluated : dict
        Dictionary including all information generated/computed from the evaluation.
    bkg_time : float
        A fraction of the total background lifetime generated during the PySTAMPAS background postprocessing. 
    save_dir : str
        Directory to save plots.

    Returns
    -------
    None.

    """
    print('\nPlotting p_lambda threshold...')
    
    # Compute number of triggers ruled as bkg/sig using xgboost and pystampas
    signalchi       = evaluated['chi'][(df_test['kind']!='bkg') & (df_test['p_lambda']!=0)]
    bkgchi          = evaluated['chi'][df_test['kind']=='bkg']
    signalp_lambda  = df_test[(df_test['kind']!='bkg') & (df_test['p_lambda']!=0)]['p_lambda']
    bkgp_lambda     = df_test[df_test['kind']=='bkg']['p_lambda']
    
    plt.figure(figsize=(15,5))
    
    # Chi threshold Plot
    plt.subplot(1,2,1) 
    plt.hist(signalchi, 1000, color='r', alpha=0.6, label='inj p_chi')
    plt.hist(bkgchi, 1000, color='b', alpha=0.6, label='bkg p_chi')
    plt.axvline(evaluated['chi_threshold'], ymax=0.9, color='k', label='threshold on p_chi')
    
    ticks = plt.xticks()[0].tolist(); ticks = [x for x in ticks if (x >= 0) and (x < 1.01) and (abs(x-evaluated['chi_threshold']) > 0.07)]
    ticks.append(round(evaluated['chi_threshold'],2)); ticks.sort()
    plt.xticks(ticks)
    plt.gca().get_xticklabels()[ticks.index(round(evaluated['chi_threshold'],2))].set_color('red') 
    
    plt.yscale('log')
    plt.title(r'$\chi$ Probability Distribution, ' + str(round((sum(signalchi <= evaluated['chi_threshold']))/len(signalchi) * 100, 2)) + '% of signal lost (<threshold)')
    plt.ylabel(r'$N$')
    plt.xlabel(r'$\chi$')
    plt.legend()
        
    # p_lambda threshold Plot
    plt.subplot(1,2,2) 
    plt.hist(signalp_lambda, 1000, color='r', alpha=0.6, label='inj p_lambda')
    plt.hist(bkgp_lambda, 1000, color='b', alpha=0.6, label='bkg p_lambda')
    plt.axvline(evaluated['p_lambda_threshold'], ymax=0.9, color='k', label='threshold on p_lambda')
    
    ticks = plt.xticks()[0].tolist(); ticks = [x for x in ticks if (((x!=0) - evaluated['p_lambda_threshold']) < 0 ) and (x >= 0) and (abs(x-evaluated['p_lambda_threshold']) > 0.6) and (x < round(np.max(signalp_lambda),1)+1)]; 
    ticks.append(round(evaluated['p_lambda_threshold'],2)); ticks.sort();
    plt.xticks(ticks)
    plt.gca().get_xticklabels()[ticks.index(round(evaluated['p_lambda_threshold'],2))].set_color('red') 
    
    plt.yscale('log')
    plt.title(r'$p_\Lambda$ Distribution, ' + str(round((sum(signalp_lambda <= evaluated['p_lambda_threshold'])) / len(signalp_lambda) * 100, 2)) + '% of signal lost (<threshold)')
    plt.xlabel(r'$p_\Lambda$')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir + '/plambda_thres.png', dpi=1000)
    
    print('p_lambda plot saved to ' + save_dir + '/plambda_thres.png')
    
def plotThresholds_L_lampda(df_test, evaluated, bkg_time, save_dir):
    """PHASE 1
    Creates xgboost and pystampas trigger distribution for xgboost probability and pystampas L_lambda
    detection statiscs, along with visual representation of thresholds set for each based off the false alarm rate chosen. 

    Parameters
    ----------
    df_test : Pandas DataFrame
        DataFrame with all data used for testing. 
    evaluated : dict
        Dictionary including all information generated/computed from the evaluation.
    bkg_time : float
        A fraction of the total background lifetime generated during the PySTAMPAS background postprocessing. 
    save_dir : str
        Directory to save plots.

    Returns
    -------
    None.

    """
    print('\nPlotting L_lambda threshold...')
    
    # Compute number of triggers ruled as bkg/sig using xgboost and pystampas
    signalchi       = evaluated['chi'][(df_test['kind']!='bkg') & (df_test['p_lambda']!=0)]
    bkgchi          = evaluated['chi'][df_test['kind']=='bkg']
    signalL_lambda  = df_test[(df_test['kind']!='bkg') & (df_test['p_lambda']!=0)]['L_lambda']
    bkgL_lambda     = df_test[df_test['kind']=='bkg']['L_lambda']
    
    plt.figure(figsize=(15,5))
    
    # Chi threshold Plot
    plt.subplot(1,2,1) 
    plt.hist(signalchi, 1000, color='r', alpha=0.6, label='inj p_chi')
    plt.hist(bkgchi, 1000, color='b', alpha=0.6, label='bkg p_chi')
    plt.axvline(evaluated['chi_threshold'], ymax=0.9, color='k', label='threshold on p_chi')
    
    ticks = plt.xticks()[0].tolist(); ticks = [x for x in ticks if (x >= 0) and (x < 1.01) and (abs(x-evaluated['chi_threshold']) > 0.07)]
    ticks.append(round(evaluated['chi_threshold'],2)); ticks.sort()
    plt.xticks(ticks)
    plt.gca().get_xticklabels()[ticks.index(round(evaluated['chi_threshold'],2))].set_color('red') 
    
    plt.yscale('log')
    plt.title(r'$\chi$ Probability Distribution, ' + str(round((sum(signalchi <= evaluated['chi_threshold']))/len(signalchi) * 100, 2)) + '% of signal lost (<threshold)')
    plt.ylabel(r'$N$')
    plt.xlabel(r'$\chi$')
    plt.legend()
    
    # L_lambda threshold Plot
    plt.subplot(1,2,2) 
    plt.hist(signalL_lambda, 1000, color='r', alpha=0.6, label='inj L_lambda')
    plt.hist(bkgL_lambda, 1000, color='b', alpha=0.6, label='bkg L_lambda')
    plt.axvline(evaluated['L_lambda_threshold'], ymax=0.9, color='k', label='threshold on L_lambda')
    
    ticks = plt.xticks()[0].tolist(); ticks = [x for x in ticks if (((x!=0) - evaluated['L_lambda_threshold']) < 0 ) and (x >= 0) and (abs(x-evaluated['L_lambda_threshold']) > 0.6) and (x < round(np.max(signalL_lambda),1)+1)]; 
    ticks.append(round(evaluated['L_lambda_threshold'],2)); ticks.sort();
    plt.xticks(ticks)
    plt.gca().get_xticklabels()[ticks.index(round(evaluated['L_lambda_threshold'],2))].set_color('red') 
    
    plt.yscale('log');
    plt.title(r'$L_\Lambda$ Distribution, ' + str(round((sum(signalL_lambda <= evaluated['L_lambda_threshold'])) / len(signalL_lambda) * 100, 2)) + '% of signal lost (<threshold)')    
    plt.xlabel(r'$L_\Lambda$')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir + '/Llambda_thres.png', dpi=400)
    
    print('L_lambda plot saved to ' + save_dir + '/Llambda_thres.png')
    
def plot_conf_matrix(xgb_cl, X, y, save_dir):
    """PHASE 1
    Plots a confusion matrix for XGBOOST

    Parameters
    ----------
    xgb_cl : sklearn.XGBCLassifier
        Training XGBoost Model
    X : Pandas DataFrame
        Data to apply the model onto. 
    y : Pandas Series
        Series containing 1's and 0's corresponding to the truth of X
    save_dir : str
        Path to save the plot

    Returns
    -------
    None.

    """
    print('\nPlotting confusion matrix diagram...')
    conf = plot_confusion_matrix(xgb_cl,
                          X,
                          y,
                          values_format='d',
                          display_labels=['Background', 'Signal'])
    conf.figure_.savefig(save_dir + '/confusion_matrix.png', dpi=400)
    print('Confusion matrix saved to ' + save_dir + '/confusion_matrix.png')
    
def plot_PR_curve(xgb_cl, X, y, save_dir, AP):
    """PHASE 1
    Plots a precision recall curve for the trained XGBoost model

    Parameters
    ----------
    xgb_cl : sklearn.XGBCLassifier
        Training XGBoost Model
    X : Pandas DataFrame
        Data to apply the model onto. 
    y : Pandas Series
        Series containing 1's and 0's corresponding to the truth of X
    save_dir : str
        Path to save the plot
    AP : float
        The average precison-recall 

    Returns
    -------
    None.

    """
    print('\nPlotting the precision-recall curve...')
    disp = plot_precision_recall_curve(xgb_cl, X, y)
    disp.ax_.set_title('Prediction-Recall Curve: Average Precision = {0:0.2f}'.format(AP))
    disp.figure_.savefig(save_dir + '/precision_recall_curve.png', dpi=400)
    print('Precision-recall plot saved to ' + save_dir + '/precision_recall_curve.png')
    
def plot_DetectionStatComparison(xgb_cl, inj, bkg, save_dir):
    """PHASE 1
    Plots a scatter/histogram comparison of the old and new detection statistics,
    p_lambda and p_chi respectively. 

    Parameters
    ----------
    inj : Pandas DataFrame
        A DataFrame containing all injection triggers.
    chi_inj : Numpy Array
        An array containing all associated injection p_chi values.
    bkg : Pandas DataFrame
        A DataFrame containg all background triggers.
    chi_bkg : Numpy Array
        An array containg all background trigger p_chi values.

    Returns
    -------
    None.

    """    
    print('\nPlotting scatter/hist detection statistic comparison...')
    
    bkg_non_zero = bkg['p_lambda']!=0
    bkg_p_lambda = bkg[bkg_non_zero]['p_lambda']
    bkg_chi = xgb_cl.predict_proba(bkg[bkg_non_zero])[:, 1]
    bkg_p_chi = bkg_chi * bkg_p_lambda
    inj_non_zero = inj['p_lambda']!=0
    inj_p_lambda = inj[inj_non_zero]['p_lambda']
    inj_chi = xgb_cl.predict_proba(inj[inj_non_zero])[:, 1]
    inj_p_chi = inj_chi * inj_p_lambda
    
    x1 = inj_p_lambda
    y1 = inj_p_chi
    x2 = bkg_p_lambda
    y2 = bkg_p_chi
    
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    
    # the scatter plot:                                                                                                                                   
    ax.scatter(x1, y1, color='r', s=0.5, alpha=0.5, label='inj')
    ax.scatter(x2, y2, color='b', s=0.5, alpha=0.5, label='bkg')
    
    # Set aspect of the main axes.                                                                                                                        
    ax.set_aspect(1.)
    
    # create new axes on the right and on the top of the current axes                                                                                     
    divider = make_axes_locatable(ax)
    # below height and pad are in inches                                                                                                                  
    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
    
    # make some labels invisible                                                                                                                          
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    
    ax_histx.hist(x1, bins=1000, color='r', alpha=0.6, label='inj p_lambda')
    ax_histy.hist(y1, bins=1000, orientation='horizontal', color='r', alpha=0.6, label='inj p_chi')
    ax_histx.hist(x2, bins=1000, color='b', alpha=0.6, label='bkg p_lambda')
    ax_histy.hist(y2, bins=1000, orientation='horizontal', color='b', alpha=0.6, label='bkg p_chi')
    
    ax_histx.set_yscale('log')
    ax_histy.set_xscale('log')
    
    ax.set_xlabel(r'$p_\Lambda$')
    ax.set_ylabel(r'$p_\chi$')
    
    ax_histx.legend(fontsize='x-small')
    ax_histy.legend(fontsize='x-small')
    ax.legend(fontsize='x-small')
    
    plt.savefig(save_dir + '/histscatter.png', dpi=1000)
    print('Plot saved to ' + save_dir + '/histscatter.png')
    
def save_a_tree(xgb_cl, save_dir):
    """PHASE 1
    Saves 1 tree created by XGBoost during training. 

    Parameters
    ----------
    xgb_cl : xgboost.XGBClassifier
        The xgboost classifier created during training.
    filename : str
        Path to save image. 

    Returns
    -------
    None.

    """
    print('\nGenerating XGBoost tree preview...')
    graph_data = xgb.to_graphviz(xgb_cl, num_trees=0, rankdir='LR')
    graph_data.format="png"
    graph_data.view(filename = save_dir + '/tree_preview')
    print('Tree saved to ' + save_dir + 'tree_preview')