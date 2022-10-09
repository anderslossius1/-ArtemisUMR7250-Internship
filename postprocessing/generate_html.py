#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions to generate HTML report pages."""

import os
import itertools
from flask import render_template
from pystampas.params import read_yaml
import flask
import numpy as np

def make_xgb_html(html_dir, xgb_report_dir, bkg_report_dir, inj_report_dir):
    """Build HTML pages to show XGBoost trainig results.
    

    Parameters
    ----------
    html_dir : str
        path to output directory for HTML pages.
    xgb_report_dir : str
        Path to report directory that contains xgb results.
    bkg_report_dir : str
        Path to report directory that contains bkg results.
    inj_report_dir : str
        Path to report directory that contains inj results.

    Returns
    -------
    None.

    """
    
    # Initiate html directory
    os.makedirs(html_dir, exist_ok=True)
    os.popen('cp templates/style.css ' + html_dir)
    os.popen('cp -r xgboost/ ' + html_dir)

    # Initiate Flask instance to generate html page from a template
    app = flask.Flask('my app')
    
    ## Results
    params_dir = xgb_report_dir + '/params'
    # BKG
    if os.path.isdir(bkg_report_dir):
        bkg_dag_infos = read_yaml(bkg_report_dir + '/analysis_params.yml')
        bkg_dag_metrics_file = '../background/' + bkg_report_dir + '/dag_metric.yml'
        timeslides = read_yaml(bkg_report_dir + '/results_stage2.yml')['Timeslides']
    else:
        bkg_dag_infos = {'Background dag info':'NA, bkg postproc was not run prior to training.'}
        bkg_dag_metrics_file = 'BKG_PREPROC_NOT_RUN'
        timeslides = 'NA, bkg postproc was not run prior to training.'
    #INJ
    if os.path.isdir(inj_report_dir):
        inj_dag_infos = read_yaml(inj_report_dir + '/analysis_params.yml')
        inj_dag_metrics_file = '../injections/' + inj_report_dir + '/dag_metric.yml'
    else:
        inj_dag_infos = {'Injection dag info':'NA, inj postproc was not run prior to training.'}
        inj_dag_metrics_file = 'INJ_PREPROC_NOT_RUN'
    num_inj = read_yaml(xgb_report_dir + '/training_report.yml')['N_INJ']
    # XGB
    tr = read_yaml(xgb_report_dir + '/training_report.yml')
    er = read_yaml(xgb_report_dir + '/evaluation/evaluation_report.yml')
    xgb_project_summary = {'Classifier name'                 : tr['CLASSIFIER_NAME'],
                           'GPS train time end'              : tr['STOP'],
                           'GPS train time start'            : tr['START'],
                           'Run'                             : tr['RUN'],
                           'Train on Fmean/Frange'           : tr['RUN_PARAMS']['FMEAN_FRANGE'],
                           'Num waveforms used for training' : tr['TRAINING/TOTAL'],
                           'FAR used for evaluation'         : er['Input False Alarm Rate'],
                           'Avg hrss@50% improvement'        : str(er['XGBoost percent HRSS@50% improvement over PySTAMPAS']) + '%'}
    params_ML = {'objective'      : tr['RUN_PARAMS']['XGB_CLASSIFIER_PARAMS']['objective'],
                 'gamma'          : tr['RUN_PARAMS']['XGB_CLASSIFIER_PARAMS']['gamma'],
                 'learning rate'  : tr['RUN_PARAMS']['XGB_CLASSIFIER_PARAMS']['learning_rate'],
                 'max tree depth' : tr['RUN_PARAMS']['XGB_CLASSIFIER_PARAMS']['max_depth'],
                 'random state'   : tr['RUN_PARAMS']['XGB_CLASSIFIER_PARAMS']['random_state']}
    
    training_waveforms = tr['TRAINING_WAVEFORMS']
    testing_waveforms = tr['TESTING_ONLY_WAVEFORMS']
    training_percentage = tr['TRAINING_PERCENTAGE'] * 100
    testing_percentage = tr['TESTING_PERCENTAGE'] * 100
    
    waveforms = er['waveforms']
    dist0 = er['dist0']
    hrss0 = er['hrss0']
    mean_freqs = er['mean_freqs']
    durations = er['durations']
    
    hrss_50_xgb = er['hrss_50_xgb']
    hrss_50_pys = er['hrss_50_pys']
    hrss_50_max = er['hrss_50_max']
    dist_50_xgb = er['dist_50_xgb']
    dist_50_pys = er['dist_50_pys']
    dist_50_max = er['dist_50_max']
    
    wvf_improv = er['Average XGBoost (over all alpha factors) detection efficiency over PySTAMPAS per waveform']
    
    # Plots
    plots_stat = [xgb_report_dir + '/evaluation/plambda_thres.png',
                  xgb_report_dir + '/evaluation/histscatter.png']
    plots_MLmets = [xgb_report_dir + '/evaluation/confusion_matrix.png',
                    xgb_report_dir + '/evaluation/precision_recall_curve.png']
    bar_improvement = xgb_report_dir + '/evaluation/improvement_bar.png'
    if os.path.isfile(xgb_report_dir + '/evaluation/Llambda_thres.png'): ## Need to finish this Llambda plot implementation
        Llambda_thres = xgb_report_dir + '/evaluation/Llambda_thres.png'

    waveform_dir = []
    waveform_dir =(np.zeros(len(waveforms), dtype='U50'))
    for i in range(len(waveforms)):
        waveform_dir[i] = 'INJ/' + waveforms[i] + '.html'
        make_xgb_waveform_html(html_dir + '/INJ', xgb_report_dir, waveforms[i])

    ## Make html
    with app.app_context():
        rendered = render_template('xgb.html',
            params_dir = params_dir,
            # BKG
            bkg_dag_infos = bkg_dag_infos,
            bkg_metrics_file = bkg_dag_metrics_file,
            timeslides = timeslides,
            # INJ
            inj_dag_infos = inj_dag_infos,
            inj_metrics_file = inj_dag_metrics_file,
            num_inj = num_inj,
            # XGB
            xgb_project_summary = xgb_project_summary,
            training_waveforms = training_waveforms,
            testing_waveforms = testing_waveforms,
            training_percentage = training_percentage,
            testing_percentage = testing_percentage,
            ML_params = params_ML,
            all_ml_params = xgb_report_dir + '/training_report.yml',
            eval_summary = dict(itertools.islice(er.items(), 2)),
            hrssimprovoverPys = er['XGBoost percent HRSS@50% improvement over PySTAMPAS'],
            plambda_metrics = er['p_lambda metrics'],
            pchi_metrics = er['p_chi metrics'],
            waveforms = waveforms,
            n_waveforms = len(waveforms),
            dist0 = dist0,
            hrss0 = hrss0,
            mean_freqs = mean_freqs,
            durations = durations,
            hrss_50_xgb = hrss_50_xgb,
            hrss_50_pys = hrss_50_pys,
            hrss_50_max = hrss_50_max,
            dist_50_xgb = dist_50_xgb,
            dist_50_pys = dist_50_pys,
            dist_50_max = dist_50_max,
            wvf_improv = wvf_improv,
            # Plots
            thres_plot = plots_stat[0],
            histscatter = plots_stat[1],
            MLmets_plots = plots_MLmets,
            improvement_bar = bar_improvement,
            waveform_dir = waveform_dir)
        
        os.makedirs(html_dir, exist_ok=True)
        html_page = html_dir + '/xgb_training.html'
        with open(html_page, 'w') as f:
            f.write(rendered)
            
    print('HTML pages generated ' + html_page)

def make_xgb_waveform_html(html_dir, xgb_report_dir, waveform):
    
    os.makedirs(html_dir, exist_ok=True)
    os.popen('cp templates/style.css ' + html_dir)
    app = flask.Flask('my app')

    curve_eff_hrss = '../' + xgb_report_dir + '/evaluation/efficiency_curves/' + waveform + '_hrss.png'
    curve_eff_dist = '../' + xgb_report_dir + '/evaluation/efficiency_curves/' + waveform + '_dist.png'

    with app.app_context():
        rendered = render_template('xgb_waveform.html', 
            waveform = waveform,
            eff_plots = [curve_eff_hrss, curve_eff_dist])
        os.makedirs(html_dir, exist_ok=True)
        with open(html_dir + '/' + waveform  + '.html', 'w') as f:
            f.write(rendered)