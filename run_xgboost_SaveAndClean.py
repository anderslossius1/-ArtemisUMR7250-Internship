"""
ONLY RUN THIS SCRIPT IF YOU HAVE COMPLETED PHASE 1 AND ARE READY TO PROCEED TO PHASE 2

THIS SCRIPT WILL:
    1. Check to make sure you have completed Phase 1 (all neccesary directories have been created) NOTE IT IS NOT PERFECT
    2. Move all of the files (dags, logs, triggers, merge triggers, reports and params) from Phase 1 to a new directory
       in the project directory called Phase 1

IF some of the files are not located for Phase 2, this script will prompt the user to either cancel (so they can backtrack 
and finish running all thats neccesary for Phase 2) or to proceed in which a 'delete all' function is called that will essentially
wipe the project directory to default. Equivalently, if the user wished to use benefit from the ladder, they could simply create
a new STAMPAS project...

@author: alossius
"""
import os
import sys
import shutil
import argparse
from pystampas.params import read_params

# Parse input
def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('xgb_params_file', type=str, help='Path to desired XGBOOST training parameters file')
    args = parser.parse_args()
    return args

def clean_project():
    """Function to wipe a project directory"""
    
    print('\nWiping project directory...')
    if os.path.isdir('dag/INJ'): shutil.rmtree('dag/INJ')
    if os.path.isdir('dag/BKG'): shutil.rmtree('dag/BKG')
    if os.path.isdir('logs'): 
        shutil.rmtree('logs')
        os.mkdir('logs')
    if os.path.isdir('report'):
        shutil.rmtree('report')
        os.mkdir('report')
    if os.path.isdir('stage1'): shutil.rmtree('stage1')
    if os.path.isdir('stage2'): shutil.rmtree('stage2')
    if os.path.isdir('inj'): shutil.rmtree('inj')
    if os.path.isdir('zerolag'): shutil.rmtree('zerolag')
    if os.path.isdir('xgboost'): shutil.rmtree('xgboost')

# Boolean variables for each neccesary directory
isINJDAG = os.path.isdir('dag/INJ')
isBKGDAG = os.path.isdir('dag/BKG')
isINJ = os.path.isdir('inj')
isST1 = os.path.isdir('stage1')
isST2 = os.path.isdir('stage2')
isXGB = os.path.isdir('xgboost')

# Dict of all the directories deemed as neccesary for Phase 2 of XGBoost
neccesary_dirs = {'dag/INJ':isINJDAG, 'dag/BKG':isBKGDAG, 'inj':isINJ, 'stage1':isST1, 'stage2':isST2, 'xgboost':isXGB}

# Asses current status of project directory
not_found = []
for path, found in neccesary_dirs.items():
    if not found:
        not_found.append(path)
if len(not_found) > 0:
    print('\nDISCLAIMER: Some of the neccesary directories for XGBoost Application (', *not_found, ') not found - XGBOOST APPLICATION WILL NOT WORK WITHOUT THEM.\n')
    print('If your intention is to proceed to Phase 2 (Application Phase), please complete Phase 1 (Training Phase) before running this script.\n')
    choice = input("If you wish to clean the current project dir anyways (erase all dags and triggers) in order to start again (from dag generation), enter 'yes'. Otherwise enter 'no' to terminate : ")
    while True:
        try:
            assert choice in ['yes', 'no']
            break
        except AssertionError:
            choice = input("Invalid entry, please type 'yes' to clean, or 'no' to exit code : ")
    if choice == 'yes':
        clean_project()
        sys.exit('project cleaned.')
    if choice == 'no':
        sys.exit('script terminated.')

# Parse input and read params
args = parse_input()
xgb_p = read_params(args.xgb_params_file, 'xgb')
inj_p = read_params(xgb_p.get('INJ_PARAMS_DIR', 'params/inj.yml'), 'inj')

# Move all Phase 1 directories
print('\nRelocating files generated in Training to /Phase1...')

save_dir = 'Phase1'
os.mkdir(save_dir)

os.mkdir(save_dir + '/dag')
os.system('mv dag/INJ/ ' + save_dir + '/dag')
os.system('mv dag/BKG/ ' + save_dir + '/dag')

os.system('mv logs ' + save_dir)
os.mkdir('logs')

os.system('mv inj ' + save_dir)
os.system('mv stage1 ' + save_dir)
os.system('mv stage2 ' + save_dir)
os.system('mv xgboost ' + save_dir)
if os.path.isdir('zerolag'): os.system('mv zerolag ' + save_dir)

os.mkdir(save_dir + '/report')
if os.path.isdir('report/bkg')      : os.system('mv report/bkg ' + save_dir + '/report')
if os.path.isdir('report/inj')      : os.system('mv report/inj ' + save_dir + '/report')
if os.path.isdir('report/stage1')   : os.system('mv report/stage1 ' + save_dir + '/report')
if os.path.isdir('report/zerolag')  : os.system('mv report/zerolag ' + save_dir + '/report')

os.system('cp -r params ' + save_dir)
os.system('cp -r postprocessing ' + save_dir)
os.system('cp -r templates ' + save_dir)

# Update HTML pages
html_path = '../' + '/'.join(xgb_p['HTML_DIR'].split('/')[3:])
phase1_html_path = html_path + '/Phase1'
os.makedirs(phase1_html_path, exist_ok=True)
os.system('mv ' + html_path + '/' + xgb_p['CLASSIFIER_NAME'] + ' ' + phase1_html_path)
if os.path.isdir(html_path + '/background') : os.system('mv ' + html_path + '/background ' + phase1_html_path)
if os.path.isdir(html_path + '/injections') : os.system('mv ' + html_path + '/injections ' + phase1_html_path)
if os.path.isdir(html_path + '/stage1')     : os.system('mv ' + html_path + '/stage1 ' + phase1_html_path)

# Restore waveforms.yml file for Phase2
old_path = inj_p.get('WAVEFORM_LIST', 'params/waveforms.yml')
new_path = old_path.replace('.yml', '_old.yml')
phase1_path = old_path.replace('.yml', '_phase1.yml')

os.rename(old_path, phase1_path)
os.rename(new_path, old_path)

# Save a Phase 1 summary for user
summary = '---THIS IS A BREIF SUMMARY FOR THIS PHASE1 (TRAINING) PROJECT---\
       \n\nTraining GPS Start time : ' + str(xgb_p['START']) + '\
         \nTraining GPS Stop time : ' + str(xgb_p['STOP']) + '\
       \n\nNumber of Injections (N_INJ) : ' + str(inj_p['INJ']['N_INJ']) + '\
       \n\nDirectories Included:\
       \n\tdag     : contains the INJ and BKG dag files created during pystampas stages \
       \n\tlogs    : contains all logs created during dag runs\
       \n\tinj     : contains all injection files and triggers\
       \n\treport  : contains bkg, inj and stage1 reports and plots generated in pystampas postprocessing\
       \n\tparams  : contains all parameter files that were used for Phase 1\
       \n\tstage1  : contains all stage1 triggers and results\
       \n\tstage2  : contains all stage2 triggers and results (background triggers)\
       \n\txgboost : contains all the training report, classifier model and evaluation files from training run\
       \n\tpostprocessing : all postprocessing functions neccesary to run the training script again if desired\
       \n\nAll the plots and summary statistics of the xgboost training run will also be on your public HTML page.\
       \n\n**IF YOU WISH, you are still capable of training a new XGBClassifier from within this sub-(phase1)-directory.\
         \nTo do this, proceed as usual by modifying the xgboost parameter file and re-running the xgboost training script.\
         \nNOTE: It is very important that you change the CLASSIFIER_NAME in the xgb.yml file. Failing to change this will cause the origal model to be erased.\
         \nLastly, you do not need to run the SaveAndClean script again to be able to use this (new) classifier, simply proceed with Phase 2 as usual, but \
           passing the name of the new classifer.\n'
            
with open(save_dir + '/Phase1_summary.txt', 'w') as f:
    f.write(summary)
    
print('Files relocated and project directory prepared for Phase 2.')
    




