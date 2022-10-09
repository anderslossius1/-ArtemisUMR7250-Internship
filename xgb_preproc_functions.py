"""
XGBOOST PRE-PROCCESSING FUNCTIONS
CONTAINS:
    PHASE 1:
    - inj_preproc_phase1() --> run_xgboost_training.py
    - bkg_preproc_phase1() --> run_xgboost_training.py
    - merge_injections() --> run_xgboost_training.py
    - merge_background() --> run_xgboost_training.py
    PHASE 2:
    - xgb_format() --> run_injections_postprocessing.py, run_background_postprocessing.py, run_zerolag_postprocessing.py
    
@author: mgorlach & alossius
"""
import os
import sys
import shutil
import numpy as np
import pandas as pd
from xgb_utils import convertFreq
from inj_functions import match_injection
from pystampas.params import read_inj_params, read_yaml
from pystampas.trigger import trigger_2det_type, trigger_3det_type
from bkg_functions import merge_triggers
inj_type = np.dtype([('alpha', float), ('start', float), ('ra', float), ('dec', float), ('iota', float), ('psi', float), ('found', bool), ('snr', float)])

def inj_preproc_phase1(sig_dir, fmr, inj_wvfs):
    """PHASE 1
    Merges and converts numpy waveform injection and trigger files to a Pandas DataFrame.
    
    Parameters
    ----------
    sig_dir : str
        Path to injection and trigger files for each injected waveform
    fmr : bool
        Boolean for whether or not to convert the frequency data to Fmean and Frange
    inj_wvfs : list
        List of all the waveforms used during the injection campaign of this project
        Built from local directory
        
    Returns
    -------
    sig : Pandas DataFrame
        DataFrame containing all neccesary information for each trigger. 
    """
    print('\nCreating injection DataFrame...')

    # Initialize
    sig_inj = pd.DataFrame() 
    sig_trg = pd.DataFrame()
    
    # Iterate over all injected waveforms
    for i, wvf in enumerate(inj_wvfs):
        # Load injection and trigger file for wvf
        inj = np.load(sig_dir + '/injections_' + wvf + '.npy')
        trg = np.load(sig_dir + '/triggers_' + wvf + '_allCut.npy')
        # Concat this wvf's inj data to sig_inj dataframe
        sig_inj = pd.concat([sig_inj, 
                             pd.DataFrame(inj)], 
                            axis=0)
        # Concat this wvf's trig data (with an additional column 'kind'=wvf) to sig_trg dataframe
        sig_trg = pd.concat([sig_trg, 
                             pd.concat([pd.DataFrame({'kind':[wvf]*len(inj)}),
                                        pd.DataFrame(trg)],
                                       axis=1)], 
                            axis=0)
        print('Progress ' + str(i+1) + '/' + str(len(inj_wvfs)) + ' ', end='\r')
    
    # Refresh indexes (precautionary step)
    sig_inj.reset_index(drop=True, inplace=True)
    sig_trg.reset_index(drop=True, inplace=True)
    
    # Concat desired aspects of sig_inj and sig_trg data + column of 1's and 0's 
    all_sig = pd.concat([sig_inj['alpha'], 
                         sig_trg.drop(['start1', 'start2', 'ifo1','ifo2','ftmap','ra','dec'], axis=1), 
                         pd.DataFrame({'Signal':(sig_trg["p_lambda"]!=0).astype(int)})], 
                        axis=1)
    
    # Convert min and max freq to mean and range freq if desired
    if fmr: all_sig = convertFreq(all_sig)
    
    print('Injection DataFrame complete.')
    return all_sig

def bkg_preproc_phase1(bkg_dir, fmr):
    """PHASE 1
    Converts a background triggers.npy file to a Pandas DataFrame.
    
    The stage2 processing often produces millions of triggers in a npy format. 
    For XGBoost, we require to convert the data into a Pandas DataFrame. However,
    due to the size of the npy file, Pandas often struggles with converting (to much memory required).
    To mitigate this a segmented approach is employed.
    
    One at a time, 1/10th of the data is converted to Pandas then saved to csv (a Pandas friendly format).
    Then, one at a time again, 1/10th of the data is read directly from csv to pandas
    and concated to a larger 'all_bkg' DataFrame. 
    
    THIS COULD BE IMPROVED, I THEORIZE IT IS NOT NECCESARY TO USE PANDAS FOR XGB.

    Parameters
    ----------
    bkg_dir : str
        Path to stage 2 merged triggers. Default is stage2/results
    fmr : bool
        Boolean for whether or not to convert the frequency data to Fmean and Frange
     
    Returns
    -------
    bkg : Pandas DataFrame
        DataFrame containing background trigger data.
    """
    print('\nCreating background DataFrame...')

    # Load merged stage2 triggers
    npy = np.load(bkg_dir + '/triggers_noCut.npy')
    
    # Define a variable corresponding to 1/10th of the amount of data
    segment = int(len(npy)/10)
    
    # Create a temp dir to store bkg data segments
    os.makedirs('xgb_temp', exist_ok=True)
    print('Temporary dir created')

    # Convert and save segmented data to Pandas then csv
    for i in range(10):
        print('Progress ' + str(i) + '/10 ', end='\r')
        bkg = pd.DataFrame(npy[:segment])
        npy = npy[segment:]
        bkg.to_csv('xgb_temp/bkg_split_' + str(i) + '.csv', index=False)
    # Clear memory
    del npy, bkg
    
    # Initialize DataFrame
    all_bkg = pd.DataFrame()
    
    # Load and concat segmented csv files
    for i in range(10):
        bkg_split = pd.read_csv('xgb_temp/bkg_split_' + str(i) + '.csv')
        all_bkg = pd.concat([all_bkg, bkg_split], axis=0) 
    # Clear memory
    del bkg_split
    
    # Remove temp directory created for segmenting
    shutil.rmtree('xgb_temp')
    print('Temporary dir removed')

    # Drop unwanted columns, concat wanted informtation, reset indexes (precautionary)
    all_bkg = all_bkg.drop(['start1', 'start2', 'ra', 'dec', 'ifo1', 'ifo2', 'ftmap'], axis=1)
    all_bkg = pd.concat([all_bkg.reset_index(drop=True), (pd.DataFrame({'kind' : ['bkg']*len(all_bkg), 'Signal' : np.zeros(len(all_bkg))})).reset_index(drop=True)], axis=1)
    all_bkg.reset_index(drop=True)
    
    # Convert min and max freq to mean and range freq if desired
    if fmr: all_bkg = convertFreq(all_bkg)
    
    print('Background DataFrame complete.')
    return all_bkg

def merge_injections(params, save_dir):
    """PHASE 1
    Function taken from run_injection_postprocessing to merge an associate inj_dag trigger files

    Parameters
    ----------
    params : dict
        injections params file.
    save_dir : str
        Directory to save merged triggers.

    Returns
    -------
    None.

    """
    print('\nAssociate and merge injections..')
    
    os.makedirs(save_dir, exist_ok=True)
    
    waveforms_dic = read_yaml(params['WAVEFORM_LIST'])
    waveforms = list(waveforms_dic.keys())
    
    ifos = params['IFOS']
    if len(ifos) == 2:
        trigger_dtype = trigger_2det_type
    elif len(ifos) == 3:
        trigger_dtype = trigger_3det_type
    
    inj_params_path = params['DAG_DIR'] + '/injection_parameters.txt'
    try:
        injections = np.genfromtxt(inj_params_path)
    except Exception as e:
        print(e)
        print('Error: could not read file ' + inj_params_path + '.')
        sys.exit(1)
        
    for wvf in waveforms:
        print('Waveform : ' + wvf)
        
        wvf_dir = params['INJ_DIR'] + '/' + wvf
        wvf_params = read_inj_params(wvf)
        alphas = os.listdir(wvf_dir)
        
        all_triggers = np.empty(0, dtype=trigger_dtype)
        
        injections_list = []
        
        for j, alpha in enumerate(alphas):
            print('Amplitude factor ' + str(j) + '/' + str(len(alphas)), end='\r')
                    
            triggers_dir = wvf_dir + '/' + alpha
            triggers = merge_triggers(triggers_dir)
            
            for i in range(len(injections)):
                network_snr = 0
                
                filename_snr = triggers_dir + '/snr_' + str(i) + '.txt'
            
                snrs = np.loadtxt(filename_snr, dtype=str)
                snrs_val = snrs[:,1].astype(float)
                
                for snr in snrs_val:
                    network_snr += snr**2
                network_snr = np.sqrt(network_snr)
                
                start = injections[i,0]
                injj = (alpha, start, injections[i,1], injections[i,2], injections[i,3],
                        injections[i,4], False, network_snr)
                injections_list.append(injj)
                duration = wvf_params['DURATION']
                stop = start + duration
                fmin = wvf_params['FMIN']
                fmax = wvf_params['FMAX']
                
                good_trigger = match_injection(triggers, start, stop, fmin, fmax)
                all_triggers = np.append(all_triggers, good_trigger)
                
        injections_list = np.array(injections_list, dtype=inj_type)
        np.save(save_dir + '/triggers_' + wvf + '_allCut', all_triggers)
        np.save(save_dir + '/injections_' + wvf, injections_list)
    
    print('Association complete. Triggers saved to ' + save_dir)

def merge_background(params, save_dir):
    """PHASE 1
    Function taken from run_background_postprocessing to merge stage2 trigger files

    Parameters
    ----------
    params : dict
        Background params file.
    save_dir : str
        Directory to save merged triggers.

    Returns
    -------
    None.

    """
    print('\nMerge background triggers from ' + params['TRIGGERS_DIR'] + '...')
    
    os.makedirs(save_dir, exist_ok=True)
    
    ifos = params['IFOS']
    n_ifos = len(ifos)
    
    triggers = merge_triggers(params['TRIGGERS_DIR'], n_dets=n_ifos)
    np.save(save_dir + '/triggers_noCut', triggers)

    print(str(len(triggers)) + ' merged and saved to ' + save_dir)

def xgb_format(trigger, p):
    """PHASE 2
    Formats the numpy void array of a PySTAMPAS trigger file for 
    the predict_proba function in inj/bkg and zerolag postprocessing scripts. 

    Parameters
    ----------
    trigger : numpy void array
        A trigger file.
    p : dict
        XGBoost params file.

    Returns
    -------
    Returns the same trigger file, but in an XGBoost friendly format for which the 
    XGBClassifier can be applied. 

    """
    # First convert void array to list, then to DataFrame
    xgbData = pd.DataFrame(trigger.tolist())
    
    # For INJ only, the shape can sometimes be along the wrong axis - so transpose if neccesary 
    if xgbData.shape == (19, 1): xgbData = xgbData.T
    
    # Obtain the names of the numpy void arrays record fields
    columns = pd.Series(np.array(trigger).dtype.names).tolist()
    
    # Append the names as columns
    xgbData.columns = columns
    
    # Match the same data format as used by inj_preproc_phase1 and bkg_preproc_phase1 for the Phase1 training.
    if p.get('FMEAN_FRANGE', True): xgbData = convertFreq(xgbData)
    
    return xgbData.drop(['start1', 'start2', 'ifo1','ifo2','ftmap','ra','dec'], axis=1).astype(float)