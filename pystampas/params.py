#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import yaml

# Comment ok

def read_yaml(fname):
    """Read a yaml file and parse it into a dictionary.
    The 'safe_load' method is used to avoid any code being executed while
    reading the yaml file.
    
    Parameters
    ----------
    fname : str
        Path to yaml file.

    Return
    ------
    dic : dict
        Dictionary built from the yaml file.
    """

    with open(fname, 'r') as f:
        dic = yaml.safe_load(f)

    return dic


def read_params(params_file, analysis_type):
    """Read parameter file for a given type of analysis.

    Parameters
    ----------
    analysis_type : str 
        Type of analysis ['preproc'/'bkg'/'inj'].
        
    Return
    ------
    full_params : dict
        A dictionary containing all the relevant parameters for the analysis.
    """


    p = read_yaml(params_file)
        
    if analysis_type == 'preproc':
        check_params(p, 'bkg')

    elif analysis_type == 'bkg':
        check_params(p, 'bkg')

    elif analysis_type == 'inj':
        check_params(p, 'inj')
        
    elif analysis_type == 'xgb':
        check_params(p, 'xgb')
    
    elif analysis_type == 'singlestage':
        print('Check params not implemented for single stage')
    else:
        print('Warning: analysis type ' + analysis_type + ' not recognized.')
        print('Parameters will not be checked')

    return p

def read_inj_params(waveform):
    """Reads parameters for the specified waveform.

    Parameters
    ----------
    waveform : str
        Name of the waveform.
    Returns
    -------
    waveform_params : dict
        Dictionary containing informations about the waveform.
    """

    inj_file = os.environ.get("STAMPASDIR") + '/waveforms/' + waveform + '.yml'
    with open(inj_file, 'r') as f:
        waveform_params = yaml.load(f, Loader=yaml.FullLoader)

    return waveform_params

    
    

# def check_bkg_params(p):
#     """
#     Check parameters dictionary for inconsistent values in time slides.
#     Raise an error if at least one test fails.

#     Parameters
#     ----------
#     p : dict
#         Parameters dictionary.
#     """
    
#     print('Check parameters relative to time-slides')
    
#     if 'DO_ZEROLAG' not in p:
#         default = False
#         print('Warning. DO_ZEROLAG not defined. Set to default: False.')
#         p['DO_ZEROLAG'] = default
    
#     assert type(p['DO_ZEROLAG']) == bool
    
#     if p['DO_ZEROLAG'] == False:
#         if 'N_LAG' not in p:
#             print('Error: N_LAG not defined while DO_ZEROLAG is False.')
#             sys.exit(1)
#         assert type(p['N_LAG']) == int
#         assert p['N_LAG'] > 0
        
#         if 'START_LAG' not in p:
#             default = 3
#             print('Warning. START_LAG (minimal time shift) not defined. Set \
# to default value of ' + str(default) + ' windows.')
#             p['START_LAG'] = default
            
#         assert type(p['START_LAG']) == int
#         if p['START_LAG'] < 2:
#             print('Warning. START_LAG is lower than 2. Risk of accidental zero-lag.')
            
#         if 'DO_MINILAGS' not in p:
#             p['DO_MINILAGS'] = False
            
#         assert type(p['DO_MINILAGS']) == bool
        
#         if p['DO_MINILAGS'] == True:
#             if 'MINILAG_DT' not in p:
#                 print('Error. MINILAG_DT not defined.')
#                 sys.exit(1)
#             assert type(p['MINILAG_DT']) in [int, float]
#             assert p['MINILAG_DT'] > np.max(p['DELTA_T'])
#             assert p['MAP_DUR'] % p['MINILAG_DT'] == 0

#     print('Check OK.')    

    

def check_params(p, analysis_type):
    """Main parameters check.

    Check that each mandatory field is defined and filled with an appropriate
    value. Return an error if it is not the case.    
    
    Parameters
    ----------
    p : dict
        Parameters dictionary.
    analysis_type : str
        Type of analysis (bkg/inj).

    Returns
    -------
    None.

    """
    
    print('Check parameters...')
    
    if analysis_type != 'xgb':      
        # Analysis configuration
        assert 'START' in p
        assert type(p['START']) in [int, float]
        assert p['START'] > 0 
        
        assert 'STOP' in p
        assert type(p['STOP']) in [int, float]
        assert p['STOP'] > p['START']
    
        assert 'RUN' in p
        assert type(p['RUN']) is str
    
        assert 'IFOS' in p
        assert type(p['IFOS']) is list
        assert len(p['IFOS']) >=2
    
        # OUTPUT DIRECTORIES
    
        assert 'DAG_DIR' in p
        assert type(p['DAG_DIR']) is str
    
        assert 'MERGED_TRIGGERS_DIR' in p
        assert type(p['MERGED_TRIGGERS_DIR']) is str
    
        assert 'HTML_DIR' in p
        assert type(p['HTML_DIR']) is str
    
        if analysis_type == 'inj':
            assert 'JOB_SIZE' in p
            assert type(p['JOB_SIZE']) is int
            assert p['JOB_SIZE'] > 0
            
            assert 'DAG_FILE_INJ' in p
            assert type(p['DAG_FILE_INJ']) is str
    
            assert 'INJ_DIR' in p
            assert type(p['INJ_DIR']) is str
    
            assert 'INJ_REPORT_DIR' in p
            assert type(p['INJ_REPORT_DIR']) is str
    
        if analysis_type == 'bkg':
            assert 'JOB_SIZE_STAGE1' in p
            assert type(p['JOB_SIZE_STAGE1']) is int
            assert p['JOB_SIZE_STAGE1'] > 0
    
            assert 'N_CLUSTER_JOB' in p
            assert type(p['N_CLUSTER_JOB']) is int
            assert p['N_CLUSTER_JOB'] > 0
    
            assert 'N_LAG_JOB' in p
            assert type(p['N_LAG_JOB']) is int
            assert p['N_LAG_JOB'] > 0
    
            assert 'DAG_FILE_STAGE1' in p
            assert type(p['DAG_FILE_STAGE1']) is str
    
            assert 'DAG_FILE_STAGE2_BKG' in p
            assert type(p['DAG_FILE_STAGE2_BKG']) is str
            
            assert 'DAG_FILE_STAGE2_ZEROLAG' in p
            assert type(p['DAG_FILE_STAGE2_BKG']) is str
    
            assert 'CLUSTERS_DIR' in p
            assert type(p['CLUSTERS_DIR']) is str
    
            assert 'MERGED_CLUSTERS_DIR' in p
            assert type(p['MERGED_CLUSTERS_DIR']) is str
    
            assert 'TRIGGERS_DIR' in p
            assert type(p['TRIGGERS_DIR']) is str
    
            assert 'MERGED_TRIGGERS_DIR' in p
            assert type(p['MERGED_TRIGGERS_DIR']) is str
    
            assert 'STAGE1_REPORT_DIR' in p
            assert type(p['STAGE1_REPORT_DIR']) is str
    
            assert 'STAGE2_REPORT_DIR' in p
            assert type(p['STAGE2_REPORT_DIR']) is str
    
        # PRE-PROCESSING
        assert 'SEGMENTS_FILE' in p
        assert type(p['SEGMENTS_FILE']) is str
    
        assert 'WINDOWS_FILE' in p
        assert type(p['WINDOWS_FILE']) is str
    
        assert 'MAP_DUR' in p
        assert type(p['MAP_DUR']) in [int, float]
        assert p['MAP_DUR'] > 0
        
        assert 'OVERLAP' in p
        assert type(p['OVERLAP']) in [int, float]
        assert p['OVERLAP'] >= 0
        assert p['OVERLAP'] < p['MAP_DUR']
        
        assert 'SAMPLE_RATE' in p
        assert type(p['SAMPLE_RATE']) in [int, float]
        assert p['SAMPLE_RATE'] > 0
        
        assert 'DO_MC' in p
        assert type(p['DO_MC']) is bool
    
        if p['DO_MC'] == True:
            assert 'PSD_FILES' in p
            for ifo in p['IFOS']:
                assert os.path.exists(os.environ['STAMPASDIR'] + '/' + p['PSD_FILES'][ifo])
    
        else:
            assert 'CALIBRATION' in p
            for ifo in p['IFOS']:
                assert ifo in p['CALIBRATION']
    
        assert 'DO_HP' in p
        assert type(p['DO_HP']) is bool
    
        if p['DO_HP'] == True:
            assert 'HP_FREQ' in p
            assert type(p['HP_FREQ']) in [int, float]
            
        assert 'DO_GATING' in p
        assert type(p['DO_GATING']) is bool
    
        if p['DO_GATING'] == True:
            assert 'GATING_THR' in p
            assert type(p['GATING_THR']) in [int, float]
            assert p['GATING_THR'] > 1
            
        assert 'DO_FREQUENCY_NOTCH' in p
        assert type(p['DO_FREQUENCY_NOTCH']) is bool
    
        if p['DO_FREQUENCY_NOTCH'] == True:
            assert 'FREQUENCY_NOTCH_FILE' in p
            assert os.path.exists(p['FREQUENCY_NOTCH_FILE'])
    
        assert 'FMIN' in p
        assert type(p['FMIN']) in [int, float]
        assert p['FMIN'] > 0 and p['FMIN'] < p['SAMPLE_RATE'] / 2
        
        assert 'FMAX' in p
        assert type(p['FMAX']) in [int, float]
        assert p['FMAX'] > p['FMIN'] and p['FMAX'] < p['SAMPLE_RATE'] / 2
    
        assert 'DELTA_T' in p
        assert type(p['DELTA_T']) is list
        for dt in p['DELTA_T']:
            assert type(dt) in [int, float]
            assert p['MAP_DUR'] % dt == 0
        
        assert 'DELTA_F' in p
        assert type(p['DELTA_F']) is list
        for df in p['DELTA_F']:
            assert type(df) in [int, float]
            assert (p['FMAX'] - p['FMIN']) % df == 0
    
        assert 'TIME_BUFFER' in p
        assert type(p['TIME_BUFFER']) is int
        assert p['TIME_BUFFER'] >= 0
        
        assert 'PSD_ESTIMATION' in p
        assert 'METHOD' in p['PSD_ESTIMATION']
        assert p['PSD_ESTIMATION']['METHOD'] in ['mean', 'mmedian', 'fullMedian']
        assert 'WINDOW' in p['PSD_ESTIMATION']
        assert type(p['PSD_ESTIMATION']['WINDOW']) is int
        assert p['PSD_ESTIMATION']['WINDOW'] > 0
        
        assert 'SAVE_FTMAPS' in p
        assert type(p['SAVE_FTMAPS']) is bool
        assert 'PREPROC_DIR' in p
        assert type(p['PREPROC_DIR']) is str
    
        # STAGE 1
        
        assert 'CLUSTERING_ALGORITHM' in p
        assert p['CLUSTERING_ALGORITHM'] in ['burstegard', 'seedless']
        
        assert 'CLUSTER_PARAMS_FILE' in p
        assert os.path.exists(p['CLUSTER_PARAMS_FILE'])
        
        assert 'SAVE_CLUSTERS' in p
        assert type(p['SAVE_CLUSTERS']) is bool
        
        if analysis_type == 'inj' and p['SAVE_CLUSTERS'] == True:
            assert 'CLUSTERS_DIR' in p
            assert type(p['CLUSTERS_DIR']) is str
            
        # STAGE 2
        
        assert 'SEARCH' in p
        assert p['SEARCH'] in ['allsky', 'targeted']
        
        assert 'USE_POLARIZED_FILTER' in p
        assert type(p['USE_POLARIZED_FILTER']) is bool
        
        assert 'N_SKY_POSITIONS' in p or 'EPSILON' in p
        if 'N_SKY_POSITIONS' in p:
            assert type(p['N_SKY_POSITIONS']) is int
            assert p['N_SKY_POSITIONS'] > 0
        if 'EPSILON' in p:
            assert type(p['EPSILON']) is float
            assert p['EPSILON'] > 0 and p['EPSILON'] < 1
            
        if p['SEARCH'] == 'targeted':
            assert 'RA' in p
            assert type(p['RA']) in [float, int]
            assert p['RA'] >= 0
            assert p['RA'] <= 24
    
            assert 'DEC' in p
            assert type(p['DEC']) in [float, int]
            assert p['DEC'] >= -90
            assert p['DEC'] <= 90
    
        assert 'DETECTION_STATISTIC' in p
        assert p['DETECTION_STATISTIC'] in ['snr_gamma', 'p_lambda', 'L_lambda', 'Lambda']
    
        # Time-slides
        if analysis_type == 'bkg':
            assert 'DO_ZEROLAG' in p
            assert type(p['DO_ZEROLAG']) is bool
    
            if p['DO_ZEROLAG'] == False:
                assert 'N_LAG' in p
                assert type(p['N_LAG']) is int
                assert p['N_LAG'] > 0
                
                assert 'START_LAG' in p
                assert type(p['START_LAG']) is int
                assert p['START_LAG'] > 0
    
                assert 'DO_MINILAGS' in p
                assert type(p['DO_MINILAGS']) is bool
                
                if p['DO_MINILAGS'] == True:
                    assert 'MINILAG_DT' in p
                    assert type(p['MINILAG_DT']) in [int, float]
                    assert p['MINILAG_DT'] > 0
                    assert p['MAP_DUR'] % p['MINILAG_DT'] == 0
                    assert p['MINILAG_DT'] > np.max(p['DELTA_T'])
                    
    
        # Injections
        if analysis_type == 'inj':
            assert 'DO_INJ' in p
            assert type(p['DO_INJ']) is bool
            
            if p['DO_INJ'] == True:
                assert 'WAVEFORM_LIST' in p
                assert os.path.exists(p['WAVEFORM_LIST'])
                assert 'INJ' in p
                assert 'TIMES' in p['INJ']
                assert 'RA' in p['INJ']
                assert 'DEC' in p['INJ']
                assert 'IOTA' in p['INJ']
                assert 'PSI' in p['INJ']
                assert 'MAX_DURATION' in p['INJ']
        
    if analysis_type == 'xgb':
        
        # START/STOP times
        assert 'START' in p
        assert type(p['START']) in [int, float]
        assert p['START'] > 0 
        
        assert 'STOP' in p
        assert type(p['STOP']) in [int, float]
        assert p['STOP'] > p['START']
    
        assert 'RUN' in p
        assert type(p['RUN']) is str
    
        assert 'IFOS' in p
        assert type(p['IFOS']) is list
        assert len(p['IFOS']) >=2
        
        # Classifier Name
        assert 'CLASSIFIER_NAME' in p
        assert type(p['CLASSIFIER_NAME']) is str
        assert p['CLASSIFIER_NAME'] is not None, 'classifier needs to be named.'
        
        # Input/Output
        assert 'INJ_PARAMS_DIR' in p
        assert type(p['INJ_PARAMS_DIR']) is str
        assert 'BKG_PARAMS_DIR' in p
        assert type(p['BKG_PARAMS_DIR']) is str
        assert 'ZL_PARAMS_DIR' in p
        assert type(p['ZL_PARAMS_DIR']) is str
        assert 'XGB_DIR' in p
        assert type(p['XGB_DIR']) is str
        assert 'XGB_PREPROC_DIR' in p
        assert type(p['XGB_PREPROC_DIR']) is str
        
        assert 'HTML_DIR' in p
        assert type(p['HTML_DIR']) is str
        
        # Data
        assert 'FMEAN_FRANGE' in p
        assert type(p['FMEAN_FRANGE']) is bool
        
        # Classifier
        assert 'XGB_CLASSIFIER_PARAMS' in p
        assert type(p['XGB_CLASSIFIER_PARAMS']) is dict
        assert 'MAX_DEPTH' in p['XGB_CLASSIFIER_PARAMS']
        assert type(p['XGB_CLASSIFIER_PARAMS']['MAX_DEPTH']) is int
        assert 'VERBOSITY' in p['XGB_CLASSIFIER_PARAMS']
        assert type(p['XGB_CLASSIFIER_PARAMS']['VERBOSITY']) is int
        assert 'REPRODUCEABLE' in p['XGB_CLASSIFIER_PARAMS']
        assert type(p['XGB_CLASSIFIER_PARAMS']['REPRODUCEABLE']) is bool
        assert 'STATE_ID' in p['XGB_CLASSIFIER_PARAMS']
        assert type(p['XGB_CLASSIFIER_PARAMS']['STATE_ID']) is int 
        
        # Training list
        assert 'TRAINING_LIST' in p
        assert type(p['TRAINING_LIST']) is dict
        assert len(p['TRAINING_LIST']) > 0
        
        if 'TRAINING_LIST_INFO' in p: assert type(p['TRAINING_LIST_INFO']) is str

        assert 'FALSE_ALARM_RATE' in p
        assert type(p['FALSE_ALARM_RATE']) in [int, float] 
        
        assert 'PLOT_LLAMBDA_THRESHOLD' in p
        assert type(p['PLOT_LLAMBDA_THRESHOLD']) is bool
        
        assert 'SAVE_XGBOOST_TREE' in p
        assert type(p['SAVE_XGBOOST_TREE']) is bool

    print('Parameters checked OK.')
    return



