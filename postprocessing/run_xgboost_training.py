"""
PHASE 1 SCRIPT - XGBOOST TRAINING

BEFORE RUNNING THIS SCRIPT:
    1. Configure inj.yml, bkg.yml and xgb.yml for training sub-project
    2. Run stage1, stage2 and inj dags
    3. (Optionally) Run background and injection postprocessing to see HTML pages

THIS SCRIPT WILL:
    1. (Optionally) merge stage2 and inj triggers
    2. Build training and evaluation DataFrames
    3. Create and train an XGBClassifier
    4. Evaluate the XGBClassifier's performance
    5. Create an HTML page on your HTML directory

AFTER THIS SCRIPT:
    1. Check the HTML pages, ensure you are happy with the preview of the model's performance
    2. Re-run script if un-happy with training parameters
    3. Once satisfied, run the SaveAndClean script to move all Phase 1 files new a new directory within the Project dir. 
    4. Re-configure inj.yml, bkg.yml and xgb.yml for application (large-scale) project (not necessary to change START/STOP in xgb.yml)
    5. Begin the Phase 2 dag generation (stage1, stage2, inj AND zerolag)

@author: alossius
"""
import os
import yaml
import shutil
import argparse
import xgboost as xgb
import pandas as pd
from datetime import datetime
from generate_html import make_xgb_html
from xgb_evaluation_functions import evaluate
from pystampas.params import read_params, read_yaml
from xgb_preproc_functions import inj_preproc_phase1, bkg_preproc_phase1, merge_injections, merge_background
from xgb_training_functions import build_training_list, split_data, train
from xgb_utils import Spinner
from xgb_plot_functions import plot_improvement_bar, plot_fitted_curves, plotThresholds_p_lambda, plotThresholds_L_lampda
from xgb_plot_functions import plot_conf_matrix, plot_PR_curve, plot_DetectionStatComparison, save_a_tree

now = datetime.now()
time_of_run = now.strftime("%b-%d-%Y %H:%M")

# Get parameter paths
def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('xgb_params_file', type=str, help='Path to desired XGBOOST training parameters file')
    args = parser.parse_args()
    return args
args = parse_input()

# Read parameters 
print('\nRead XGB parameters file ' + args.xgb_params_file)
xgb_p = read_params(args.xgb_params_file, 'xgb') 
print('\nRead INJ parameters file ' + xgb_p.get('INJ_PARAMS_DIR', 'params/inj.yml'))
inj_p = read_params(xgb_p.get('INJ_PARAMS_DIR', 'params/inj.yml'), 'inj')
print('\nRead BKG parameters file ' + xgb_p.get('BKG_PARAMS_DIR', 'params/bkg.yml'))
bkg_p = read_params(xgb_p.get('BKG_PARAMS_DIR', 'params/bkg.yml'), 'bkg')
print('\nRead BKG Stage 2 results file ' + bkg_p.get('STAGE2_REPORT_DIR', 'ERROR: EMPTY') + '/results_stage2.yml')
bkg_r = read_yaml(bkg_p.get('STAGE2_REPORT_DIR', 'report/bkg') + '/results_stage2.yml')

classifier_name = xgb_p.get('CLASSIFIER_NAME', time_of_run)
print('\n--------------START XGBOOST < ' + classifier_name + ' > TRAINING--------------')

# Obtain waveforms used for injections
waveforms_dic = read_yaml(inj_p['WAVEFORM_LIST'])
injected_waveforms = list(waveforms_dic.keys())

# Create report directories
xgb_dir = xgb_p.get('XGB_DIR', 'xgboost') + '/' + classifier_name
xgb_preproc_dir = xgb_p.get('XGB_PREPROC_DIR', 'xgboost/preproc')
xgb_evaluation_dir = xgb_p.get('XGB_DIR', 'xgboost') + '/' + classifier_name + '/evaluation'

os.makedirs(xgb_dir, exist_ok=True)
os.makedirs(xgb_evaluation_dir, exist_ok=True)

# Format data for XGBoost
inj_data_path = inj_p.get('MERGED_TRIGGERS_DIR', 'inj/results')
bkg_data_path = bkg_p.get('MERGED_TRIGGERS_DIR', 'stage2/results')

if os.path.isdir(xgb_preproc_dir):
    print('\nReading in DataFrames...')
    with Spinner(): 
        inj = pd.read_csv(xgb_preproc_dir + '/df_inj.csv')
        bkg = pd.read_csv(xgb_preproc_dir + '/df_bkg.csv')
    print('INJ and BKG data read.')
else:
    if not os.path.isdir(inj_data_path):
        merge_injections(inj_p, inj_data_path)
    if not os.path.isdir(inj_data_path):
        merge_background(bkg_p, bkg_data_path)
    os.makedirs(xgb_preproc_dir, exist_ok=True) 
    inj = inj_preproc_phase1(inj_data_path, xgb_p.get('FMEAN_FRANGE', True), injected_waveforms)
    bkg = bkg_preproc_phase1(bkg_data_path, xgb_p.get('FMEAN_FRANGE', True))
    print('\nSaving DataFrames...')
    try: 
        with Spinner(): 
            inj.to_csv(xgb_preproc_dir + '/df_inj.csv', index=False)
            bkg.to_csv(xgb_preproc_dir + '/df_bkg.csv', index=False) 
    except Exception as e:
        print('DataFrames could not be saved: ' + e)
        shutil.rmtree(xgb_preproc_dir)
    print('DataFrames saved to /' + xgb_preproc_dir)

# Build training list
train_list = build_training_list(xgb_p, injected_waveforms)
test_only_list = [wvf for wvf in injected_waveforms if wvf not in train_list]

# Split data
training_percentage = len(train_list)/len(injected_waveforms)
testing_percentage = 1 - training_percentage
debug_options = [xgb_p['XGB_CLASSIFIER_PARAMS']['REPRODUCEABLE'], xgb_p['XGB_CLASSIFIER_PARAMS']['STATE_ID']]

df_train, df_test = split_data(inj, bkg, train_list, training_percentage, debug_options)

# Train or load in model
if os.path.isfile(xgb_dir + '/XGBClassifier.json'):
    print('Read in classifier from ' + xgb_dir + '/XGBClassifier.json')
    xgb_cl = xgb.XGBClassifier()
    xgb_cl.load_model(xgb_dir + '/XGBClassifier.json')
else:
    classifier_params = {}
    for k,v in xgb_p['XGB_CLASSIFIER_PARAMS'].items(): classifier_params[k] = v
    
    with Spinner(): xgb_cl = train(df_train, classifier_params)

    xgb_cl.save_model(xgb_dir + '/XGBClassifier.json')
    print('Classifier saved to : ' + xgb_dir + '/XGBClassifier.json')

# Copy parameter files into the report directory
os.popen('cp -r params ' + xgb_dir)

# Save training report
with open(xgb_dir + '/training_report.yml', 'w') as f:
    yaml.dump({'CLASSIFIER_NAME':classifier_name,
               'START':xgb_p['START'],
               'STOP':xgb_p['STOP'],
               'RUN': xgb_p['RUN'],
               'IFOS': xgb_p['IFOS'],
               'N_INJ':inj_p['INJ']['N_INJ'],
               'TRAINING/TOTAL':str(len(train_list)) + '/' + str(len(injected_waveforms)),
               'TRAINING_WAVEFORMS':train_list,
               'TRAINING_PERCENTAGE':round(0.75*training_percentage,2),
               'TESTING_PERCENTAGE':round(testing_percentage + 0.25*training_percentage,2),
               'TESTING_ONLY_WAVEFORMS':test_only_list,
               'TRAINING_LIST_DETAILS':xgb_p['TRAINING_LIST_INFO'],
               'RUN_PARAMS':{'FMEAN_FRANGE':xgb_p['FMEAN_FRANGE'],
                             'XGB_CLASSIFIER_PARAMS':xgb_cl.get_params()}}, 
              f, sort_keys=False)

print('Training report saved to : ' + xgb_dir + '/training_report.yml')

# Run classifier evaluation
bkg_time = (testing_percentage + 0.25*training_percentage) * bkg_r['Background lifetime'] 

#COULD OPT TO EXTERNALLY SAVE 'EVALUATED' AND THEN CONDITIONALY RUN THE EVAL FUNCTION ONLY IF IT HASNT BEEN RUN PRIOR - SAVE TIME, NOT MANDATORY.
evaluated = evaluate(df_test, xgb_cl, xgb_p['FALSE_ALARM_RATE'], bkg_time, inj_p, injected_waveforms)

# Create diagnostic plots
plot_conf_matrix(xgb_cl, evaluated['X'], evaluated['y'], xgb_evaluation_dir)
plot_PR_curve(xgb_cl, evaluated['X'], evaluated['y'], xgb_evaluation_dir, evaluated['average_precision'])
plot_improvement_bar(evaluated, train_list, xgb_evaluation_dir)
plot_fitted_curves(evaluated, 'hrss', xgb_evaluation_dir)
plot_fitted_curves(evaluated, 'dist', xgb_evaluation_dir)
plotThresholds_p_lambda(df_test, evaluated, bkg_time, xgb_evaluation_dir)
if xgb_p.get('PLOT_LLAMBDA_THRESHOLD', False): plotThresholds_L_lampda(df_test, evaluated, bkg_time, xgb_evaluation_dir)
plot_DetectionStatComparison(xgb_cl, inj.drop(['kind', 'Signal', 'alpha'], axis=1), bkg.drop(['kind', 'Signal'], axis=1), xgb_evaluation_dir)
# if xgb_p.get('SAVE_XGBOOST_TREE', False): save_a_tree(xgb_cl, xgb_evaluation_dir) # CANNOT BE DONE UNTIL GRAPHIZ IS INSTALLED INTO CALTECH CLUSTER

# Save evaluation report
with open(xgb_evaluation_dir + '/evaluation_report.yml', 'w') as f:
    yaml.dump({'Input False Alarm Rate':xgb_p['FALSE_ALARM_RATE'],
               'Nearest possible FAR':(evaluated['far']),
               'sklearn Average Precision':evaluated['average_precision'], 
               'sklearn AUC_PR':evaluated['auc_precision_recall'],
               'Prevalence of Data (P/P+N)':evaluated['prevalence'],
               'XGBoost percent HRSS@50% improvement over PySTAMPAS':round(evaluated['frac_of_maximum']*100,1),
               'p_lambda metrics':{'Num True Positives':evaluated['pys_all_truepos'],
                                    'Num True Negatives':evaluated['pys_all_trueneg'],
                                    'Num False Positives':evaluated['pys_all_falspos'],
                                    'Num False Negatives':evaluated['pys_all_falsneg'],
                                    'Precision':evaluated['pys_precision'], 
                                    'Recall':evaluated['pys_recall'], 
                                    'False Positive Rate':evaluated['pys_false_positive_rate'], 
                                    'False Alarm Rate':evaluated['pys_false_alarm_rate']},
               'p_chi metrics':{'Num True Positives':evaluated['xgb_all_truepos'],
                                    'Num True Negatives':evaluated['xgb_all_trueneg'],
                                    'Num False Positives':evaluated['xgb_all_falspos'],
                                    'Num False Negatives':evaluated['xgb_all_falsneg'],
                                    'Precision':evaluated['xgb_precision'], 
                                    'Recall':evaluated['xgb_recall'], 
                                    'False Positive Rate':evaluated['xgb_false_positive_rate'], 
                                    'False Alarm Rate':evaluated['xgb_false_alarm_rate']},
               'Average XGBoost (over all alpha factors) detection efficiency over PySTAMPAS per waveform':evaluated['wvf_eff_improv'],
               'dist0':evaluated['dist0'],
               'hrss0':evaluated['hrss0'],
               'mean_freqs':evaluated['mean_freqs'],
               'durations':evaluated['durations'],
               'hrss_50_xgb':evaluated['hrss_50_xgb'],
               'hrss_50_pys':evaluated['hrss_50_pys'],
               'hrss_50_max':evaluated['hrss_50_max'],
               'dist_50_xgb':evaluated['dist_50_xgb'],
               'dist_50_pys':evaluated['dist_50_pys'],
               'dist_50_max':evaluated['dist_50_max'],
               'waveforms':injected_waveforms}, 
              f, sort_keys=False)
    
print('Classifier evaluation report saved to ' + xgb_evaluation_dir + '/evaluation_report.yml')

# Generate XGBoost HTML Docs
print('\nGeneratimg HTML report pages...')
html_dir = xgb_p['HTML_DIR'] + '/' + classifier_name
bkg_dir = bkg_p.get('STAGE2_REPORT_DIR', 'report/bkg')
inj_dir = inj_p.get('INJ_REPORT_DIR', 'report/inj')
make_xgb_html(html_dir, xgb_dir, bkg_dir, inj_dir)

print('\n--------------END XGBOOST < ' + classifier_name + ' > TRAINING--------------')
