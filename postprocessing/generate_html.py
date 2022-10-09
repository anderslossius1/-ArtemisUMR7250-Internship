#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions to generate HTML report pages."""

import os
import glob
import itertools
from flask import render_template
from pystampas.params import read_yaml
import flask
import yaml
import numpy as np

# Structure of html directory :
# Project/
    # /H1L1
    # /V1H1...
        # bkg.html
        # inj.html
        # zerolag.html
        # style.css
        # BKG/ + vetos.html
        # INJ/ + waveforms.html
        # ZEROLAG/ + vetos.html
        # report/ + all plots and data

def set_html_dir(html_dir):
    """Set-up html directory: create directory and copy report folder in it."""
    
    os.makedirs(html_dir, exist_ok=True)
    os.popen('cp templates/style.css ' + html_dir)
    os.popen('cp -r report ' + html_dir)



def make_dq_html(html_dir, ifo):
    """Make HTML pages to summarize DQ flags"""
    set_html_dir(html_dir)
    os.makedirs(html_dir + '/DQ_' + ifo, exist_ok=True)

    app = flask.Flask('my app')

    report_dir = 'report/DQ/' + ifo 
    flags = os.listdir(report_dir)
    flags_html = ['DQ_' + ifo + '_' + fl + '.html' for fl in flags]
    # Generate report HTML page for each flag
    for fl in flags:
        flag_dir = report_dir + '/' + fl
        flag_html = html_dir +  '/DQ_' + ifo + '_' + fl + '.html'
        make_single_dq_html(fl, flag_dir, flag_html, ifo)
        
    # Generate summary HTML page
    n_flags = len(flags)
    with app.app_context():
        rendered = render_template('DQ_index.html', \
        det = ifo,
        flags = flags,
        links = flags_html,
        n = n_flags)                               
    html_page = html_dir + '/DQflags_' + ifo + '.html'
    with open(html_page, 'w') as f:
        f.write(rendered)

def make_single_dq_html(flag, flag_dir, flag_html, ifo):
    """Make HTML page that summarize one DQ flag."""
    
    app = flask.Flask('my app')
    with open(flag_dir + '/summary_' + flag + '.yml', 'r') as f:
        summary = yaml.load(f, Loader=yaml.FullLoader)
    
    plot = flag_dir + '/clusters_distribution_' + flag + '.png'
    loudest = np.load(flag_dir + '/loudest_' + flag + '.npy')
    with app.app_context():
        rendered = render_template('DQ_individual.html', \
            flag = flag,
            summary = summary, \
            plot = plot,\
            clusters = loudest,\
            det = ifo,
            n_loudest = len(loudest))
                                               
        html_page = flag_html
        with open(html_page, 'w') as f:
            f.write(rendered)


def make_background_html(html_dir, report_dir):
    """Build HTML pages to show background estimation results.
    
    Parameters
    ----------
    html_dir : str
        path to output directory for HTML pages.
    report_dir : str
        Path to report directory that contains the results.

    Returns
    -------
    None.

    """
    
    # Initiate html directory
    os.makedirs(html_dir, exist_ok=True)
    os.popen('cp templates/style.css ' + html_dir)
    os.popen('cp -r report/ ' + html_dir)

    # Initiate Flask instance to generate html page from a template
    app = flask.Flask('my app')

    # Get results to feed the HTML page
    dag_infos = read_yaml(report_dir + '/analysis_params.yml')
    stage2_infos = read_yaml(report_dir + '/results_stage2.yml')
    vetos = read_yaml(report_dir + '/vetos.yml')
    
    far_plots = glob.glob(report_dir + '/FAR/*.png')
    plots_stats = glob.glob(report_dir + '/main_plots/statistics/*.png')
    plots_morpho = glob.glob(report_dir + '/main_plots/morphology/*.png')
    plots_tf = glob.glob(report_dir + '/main_plots/timefrequency/*.png')
    plots_sky = glob.glob(report_dir + '/main_plots/sky/*.png')

    params_dir = report_dir + '/params' #e.g report/stage1/params
    dag_metrics_file = report_dir + '/dag_metric.yml'

    loudest_triggers = np.load(report_dir + '/loudest/loudest_triggers_allCut.npy')
    falsealarm = np.load(report_dir + '/loudest/falsealarm.npy')
    far = falsealarm[:,0]
    fap = falsealarm[:,1]
    ftmaps = sorted(glob.glob(report_dir + '/loudest/trigger*.png'))
    
    # Manage vetos
    vetos_plots = {}
    for veto in vetos.keys():
        if veto == 'noCut':
            continue
        veto_plots_dir = report_dir + '/' + veto
        vetos_plots[veto] = veto_plots_dir
    # Generate html page from template
    with app.app_context():
        rendered = render_template('bkg.html', \
            dag_infos = dag_infos, \
            stage2_infos = stage2_infos,
            far_plots = far_plots,
            stat_plots = plots_stats,
            morphology_plots = plots_morpho,
            tf_plots = plots_tf,
            sky_plots = plots_sky,
            params_dir = params_dir,
            metrics_file = dag_metrics_file,
            n_loudest = len(loudest_triggers),
            clusters = loudest_triggers,
            far = far,
            fap = fap,
            ftmaps = ftmaps,
            vetos = vetos_plots)
                                   
        os.makedirs(html_dir, exist_ok=True)
        html_page = html_dir + '/background.html'
        with open(html_page, 'w') as f:
            f.write(rendered)

def make_zerolag_html(html_dir, report_dir, ifos):
    '''Generates html report pages for zero-lag analysis.'''

    # Initiate html directory
    set_html_dir(html_dir, ifos)
    os.makedirs(html_dir + '/' + ifos + '/BKG', exist_ok=True)

    # Initiate Flask instance to generate html page from a template
    app = flask.Flask('my app')

    # Load informations files
    with open(report_dir + '/dag_infos.yml', 'r') as f:
        dag_infos = yaml.load(f, Loader=yaml.FullLoader)

    with open(report_dir + '/results_zl.yml', 'r') as f:
        stage2_infos = yaml.load(f, Loader=yaml.FullLoader)

    # Generate main report page
    far_plot = 'report/zerolag/FARs/far_zerolag.png'
    diagnostic_plots = glob.glob(report_dir + '/diagnostic/*.png')
    loudest_triggers = np.load(report_dir + '/loudest/loudest_triggers_allCut.npy')
    falsealarm = np.load(report_dir + '/loudest/falsealarm.npy')
    far = falsealarm[:,0]
    fap = falsealarm[:,1]
    ftmaps = glob.glob(report_dir + '/loudest/trigger*.png')
    # Generate a html page from a template
    with app.app_context():
        rendered = render_template('zerolag.html', \
            dag_infos = dag_infos, \
            stage2_infos = stage2_infos,
            far_plot = far_plot,
            diagnostic_plots = diagnostic_plots,
            n_loudest = len(loudest_triggers),
            clusters = loudest_triggers,
            far = far,
            fap = fap,
            ftmaps=ftmaps)
        os.makedirs(html_dir, exist_ok=True)
        with open(html_dir + '/' + ifos + '/zerolag.html', 'w') as f:
            f.write(rendered)

    print('html pages generated.')

##-- STAGE 1 --##

def make_stage1_html(html_dir, report_dir, ifo):
    """
    Generate HTML report page for STAGE 1 and a given detector. 

    Parameters
    ----------
    html_dir : str
        Path to output html directory (e.g ../public_html/MyProject).
    report_dir : str
        Path to input report directory.
    ifo : str
        Detector name.

    Returns
    -------
    None.

    """

    # Initiate html directory
    os.makedirs(html_dir, exist_ok=True)
    os.popen('cp templates/style.css ' + html_dir)
    os.popen('cp -r report/ ' + html_dir)
    os.makedirs(html_dir + '/STAGE1/' + ifo, exist_ok=True)

    # Initiate Flask instance to generate html page from a template
    app = flask.Flask('my app')

    # Get basic info about the project
    dag_infos = read_yaml(report_dir + '/analysis_params.yml')

    # Read stage 1 results and plots
    stage1_infos = read_yaml(report_dir + '/results_stage1_' + ifo + '.yml')

    diagnostic_plots = glob.glob(report_dir + '/' + ifo + '/diagnostic/*.png')
    loudest_clusters = np.load(report_dir + '/' + ifo + '/loudest/loudest_clusters_allCut.npy')

    params_dir = report_dir + '/params' #e.g report/stage1/params
    # Generate html page from template
    with app.app_context():
        rendered = render_template('stage1.html', \
            det = ifo,
            dag_infos = dag_infos, \
            stage1_infos = stage1_infos,
            diagnostic_plots = diagnostic_plots,
            n_loudest = len(loudest_clusters),
            clusters = loudest_clusters,
            params_dir = params_dir)
                                   
        os.makedirs(html_dir, exist_ok=True)
        with open(html_dir + '/stage1_' + ifo + '.html', 'w') as f:
            f.write(rendered)

    print('html pages generated.')

def make_inj_html(html_dir, report_dir):

    # Initiate html directory
    os.makedirs(html_dir, exist_ok=True)
    os.popen('cp templates/style.css ' + html_dir)
    os.popen('cp -r report/ ' + html_dir)

    # Initiate Flask instance to generate html page from a template
    app = flask.Flask('my app')

    # Get results to feed the HTML page
    dag_infos = read_yaml(report_dir + '/analysis_params.yml')
    vetos = read_yaml(report_dir + '/vetos.yml')
    params_dir = report_dir + '/params' #e.g report/stage1/params
    dag_metrics_file = report_dir + '/dag_metric.yml'

    hrss0 = []
    dist0 = []
    hrss50 = []
    dist50 = []
    hrss90 = []
    dist90 = []
    waveform_dir = []
    mean_freqs = []
    durations = []
    
    results = np.genfromtxt(report_dir + '/results.txt', dtype=str, encoding='utf-8')
    results = np.atleast_2d(results) # in case there is only one waveform
    waveforms = results[:,0]
    hrss0 =(results[:,1])
    dist0 =(results[:,2])
    hrss50 =(results[:,3])
    dist50 =(results[:,4])
    hrss90 =(results[:,5])
    dist90 =(results[:,6])
    mean_freqs =(results[:,7])
    durations =results[:,8]
    waveform_dir =(np.zeros(len(waveforms), dtype='U50'))
    
    # Generate specific html page for each waveform
    for i in range(len(waveforms)):
        waveform_dir[i]  =  'INJ/' + waveforms[i] + '.html'
        make_waveform_html(html_dir + '/INJ', report_dir, waveforms[i])

    with app.app_context():

        rendered = render_template('inj.html', \
            dag_infos = dag_infos, 
            n_waveforms = len(waveforms),
            waveforms = waveforms,
            hrss0 = hrss0,
            dist0 = dist0,
            hrss50 = hrss50,
            dist50 = dist50,
            hrss90 = hrss90,
            dist90 = dist90,
            params_dir = params_dir,
            metrics_file = dag_metrics_file,
            waveform_dir = waveform_dir,
            vetos = vetos,
            mf = mean_freqs,
            dur = durations)
        os.makedirs(html_dir, exist_ok=True)

        with open(html_dir + '/inj.html', 'w') as f:
            f.write(rendered)

    print('html pages generated.')

def make_waveform_html(html_dir, report_dir, waveform):
    '''Generates html report page for a specific waveform.'''

    app = flask.Flask('my app')
    dag_infos = read_yaml(report_dir + '/analysis_params.yml')


    results = np.genfromtxt(report_dir + '/results.txt', dtype=str, encoding='utf-8')
    results = np.atleast_2d(results) # in case there is only one waveform
    waveforms = results[:,0]
    hrss0 = (results[:,1])
    dist0 = (results[:,2])
    hrss50 = (results[:,3])
    dist50 = (results[:,4])
    i = np.where(waveforms==waveform)[0][0]

    efficiency = {'Nominal hrss' : hrss0[i],
                  'Nominal Distance (Mpc)' : dist0[i],
                  'hrss @50%' : hrss50[i],
                  'Distance @50% (Mpc)' : dist50[i],
                  }
    diagnostic_plots = glob.glob(report_dir + '/' + waveform + '/diagnostic/*')
    efficiency_plots = glob.glob(report_dir + '/'  + waveform + '/efficiency/*')
    diagnostic_plots = ['../' + path for path in diagnostic_plots]
    efficiency_plots = ['../' + path for path in efficiency_plots]
    with app.app_context():

        rendered = render_template('waveform.html', \
            waveform = waveform,
            dag_infos = dag_infos, \
            efficiency = efficiency,
            efficiency_plots = efficiency_plots,
            diagnostic_plots = diagnostic_plots)
        os.makedirs(html_dir, exist_ok=True)
        with open(html_dir + '/' + waveform  + '.html', 'w') as f:
            f.write(rendered)

def make_bkg_cut_html(html_dir, cut, ifos):

    app = flask.Flask('my app')

    diagnostic_plots = glob.glob('report/stage2/diagnostic/' + cut + '/*.png')
    far_plot = '../report/stage2/FARs/far_' + ifos + '_' + cut + '.png'
    diagnostic_plots = ['../' + path for path in diagnostic_plots]
    with app.app_context():

        rendered = render_template('bkg_cut.html', \
            cut = cut,
            far_plot = far_plot,
            diagnostic_plots = diagnostic_plots)
        os.makedirs(html_dir, exist_ok=True)
        with open(html_dir + '/' + cut  + '.html', 'w') as f:
            f.write(rendered)

def make_stage1_cut_html(html_dir, cut, ifo):

    app = flask.Flask('my app')

    diagnostic_plots = glob.glob('report/stage1/' + ifo + '/diagnostic/' + cut + '/*.png')
    diagnostic_plots = ['../../' + path for path in diagnostic_plots]
    with app.app_context():

        rendered = render_template('stage1_cut.html', \
            ifo = ifo,
            cut = cut,
            diagnostic_plots = diagnostic_plots)
        os.makedirs(html_dir, exist_ok=True)
        with open(html_dir + '/' + ifo + '_' + cut  + '.html', 'w') as f:
            f.write(rendered)
            
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