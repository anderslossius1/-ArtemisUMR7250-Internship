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
        
    Credit @Adrian Macquet, Artemis UMR 7250.    
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
        
    Credit @Adrian Macquet, Artemis UMR 7250.
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


def check_params(p, analysis_type):
    """ Function to check parameter files.
    """
        
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



