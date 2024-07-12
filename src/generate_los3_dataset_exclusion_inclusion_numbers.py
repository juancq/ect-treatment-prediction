import argparse
import logging
import re
import yaml
import pandas as pd
import numpy as np
import random
from collections import OrderedDict

from survival_utils_ect import build_re_patterns
from survival_utils_ect import filter_transfers
from survival_utils_ect import get_pt_features
from survival_utils_ect import charlson
from tqdm import tqdm


# for profiling
#import cProfile, pstats, io


any_psych_re = re.compile('F(?:2[0-9]|3[0-9])')
def check_any_psych(df):
    df_flat = df.values.flatten()
    return np.array([bool(any_psych_re.search(i)) for i in df_flat]).any() 
                

ECT_CODES = '14224-0[0-6]'
ect_re = re.compile(ECT_CODES)
def check_ect(df):
    df_flat = df.values.flatten()
    return np.array([bool(ect_re.search(i)) for i in df_flat]).any() 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_data', metavar='episode_data', type=str, nargs='+',
                        help='patient chunks of episode data from APDC')
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--dropna', action='store_true')
    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()
    if args.config:
        with open(args.config) as fin:
            config = yaml.full_load(fin)

    logging.basicConfig(filename='los3_exclusion_log.txt', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)

    diagnosis_cols = ['diagnosis_codeP'] + [f'diagnosis_code{i}' for i in range(1,51)]
    procedure_cols = ['procedure_codeP'] + [f'procedure_code{i}' for i in range(1,51)]

    nrows = None

    if args.all:
        usecols = None
    else:
        usecols = ['episode_start_date', 'ppn', 'episodeset', 
                    'hospital_type', 'SEX', 'EMERGENCY_STATUS_RECODE', 'PAYMENT_STATUS_ON_SEP',
                    'SA2_2016_CODE',
                    'age_recode',
                    'block_numP',
                    'age_grouping_recode',
                    'MARITAL_STATUS',
                    'INVOLUNTARY_DAYS_IN_PSYCH',
                    'DAYS_IN_PSYCH_UNIT',
                    'MDC',
                    'EPISODE_LENGTH_OF_STAY',
                    'totlos',
                    'apdc_project_recnum',
                    'firstadm', 'finsep',
                    'EPISODE_LEAVE_DAYS_TOTAL', 
                    ]
        usecols.extend(diagnosis_cols)
        usecols.extend(procedure_cols)

    
    # regular expressions for diagnosis and procedures of interest
    diagnosis_procedure_re = build_re_patterns()

    # exclusion stats
    exclusion_criteria = ['lookback', 'under18', 'death_at_index', 'no_primary_psych',
                            'prior_ect', 'any_psych', 'ect_at_index', 
                            'ect_date_error']
    exclusion = {e: 0 for e in exclusion_criteria}

    STUDY_START_DATE = pd.to_datetime(config['start_date'])
    date_lookback = pd.offsets.DateOffset(months=config['lookback_months'])
    ect_counter = 0
    mh_amb_counter = 0
    PT_GROUPS = 20
    total = 0
    for i in tqdm(range(PT_GROUPS)):
        fname = config['data'].replace('*', str(i))
        mh_fname = config['mh_amb'].replace('*', str(i))
        logging.info(f'Processing {fname}')
        print(fname)
        # read apdc
        df = pd.read_csv(fname, nrows=nrows, usecols=usecols, dtype='str', encoding='ISO-8859-1')
        df = df.astype(str)

        df['age_recode'] = df['age_recode'].astype('float')
        df.loc[:,'age_recode'] = df['age_recode'].fillna(0).astype(float)
        df_sex = pd.get_dummies(df, columns=['SEX'])
        if 'SEX_Indeterminate' not in df_sex.columns:
            df_sex['SEX_Indeterminate'] = 0
        sex_cols = ['SEX_Female', 'SEX_Male', 'SEX_Indeterminate']
        df[sex_cols] = df_sex[sex_cols].astype(int)
        df['episodeset'] = df['episodeset'].astype('int')
        
        time_cols = ['DAYS_IN_PSYCH_UNIT', 'EPISODE_LEAVE_DAYS_TOTAL', 
                    'INVOLUNTARY_DAYS_IN_PSYCH', 'EPISODE_LENGTH_OF_STAY']
        df[time_cols] = df[time_cols].astype('float')
           
        df['start_date'] = pd.to_datetime(df['firstadm'], format='%d%b%Y')
        df['year'] = df['start_date'].apply(lambda x: x.year)

        patient_group = df.groupby('ppn')
        print('Done grouping', flush=True)

        total += len(patient_group)
        continue

        records = []
        for ppn,group in tqdm(patient_group, miniters=500, maxinterval=5, position=0, leave=True):

            episodes_group = group.groupby('episodeset')
            # condense transfers into a single episode (saving all diagnoses and procedures)
            episodes = episodes_group.apply(filter_transfers).reset_index(drop=True)
            episodes.sort_values('episodeset', inplace=True)

            study_group = episodes
            # look for primary psychiatric diagnosis
            psych_mask = study_group['diagnosis_codeP'].str.contains('F(?:2[0-9]|3[0-9])', na=False)
            psych_episodes = study_group[psych_mask]

            # keep those with length of stay less than 3 days
            los_mask = psych_episodes['EPISODE_LENGTH_OF_STAY'] >= 3
            psych_episodes = psych_episodes[los_mask]

            # this should never happen because of how raw files were created
            if psych_episodes.empty: 
                exclusion['no_primary_psych'] += 1
                continue

            patient_exclusions = OrderedDict()
            # collect episodes that meet inclusion criteria
            matching_episodes = []
            for index, index_admission in psych_episodes.iterrows():  
                if index_admission.age_recode < 18:
                    patient_exclusions['under18'] = None
                    continue
                # exclude those with not enough lookback 
                elif index_admission.start_date < STUDY_START_DATE+date_lookback:
                    patient_exclusions['lookback'] = None
                    continue
    
                # medical history without index admission
                history_no_index = study_group[(study_group.start_date < index_admission.start_date) & (study_group.start_date >= index_admission.start_date - date_lookback)]
                # exclude patients with any severe psychiatric diagnosis during lookback period
                if check_any_psych(history_no_index[diagnosis_cols]):
                    patient_exclusions['any_psych'] = None
                    continue
    
                # exclude patients with ECT prior to index psych
                if check_ect(history_no_index[procedure_cols]):
                    patient_exclusions['prior_ect'] = None
                    continue
    
                # episode matches all exclusion criteria
                matching_episodes.append(index_admission.to_dict())
            
            # no episodes that match exclusion criteria 
            if matching_episodes == []:
                # keep just the first exclusion criteria
                for key in patient_exclusions.keys():
                    exclusion[key] += 1
                    break
                continue

            # select at random one of the matching episodes to be the index admission
            index_admission = pd.Series(random.choice(matching_episodes))

            if check_ect(index_admission[procedure_cols]):
                ect_counter += 1
            

    logging.info('Exclusion criteria')
    logging.info(f'total N: {total}')
    return
    logging.info(f'{name} {value}')
    for name, value in exclusion.items():
        logging.info(f'{name} {value}')
    logging.info(f'Finished processing {fname}')
    print(f'ect counter {ect_counter}')


if __name__ == '__main__':
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #ps = pstats.Stats(pr).sort_stats('cumtime')
    #ps.print_stats()
