import argparse
import logging
import re
import yaml
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import random

from survival_utils_ect import build_re_patterns, build_re_patterns_from
from survival_utils_ect import filter_transfers
from survival_utils_ect import get_pt_features
from survival_utils_ect import charlson
from survival_utils_ect import get_outcomes, get_death_reason
from tqdm import tqdm


# for profiling
#import cProfile, pstats, io

def check_death(ppn: str, death_index: pd.DataFrame, drop_duplicates: bool = False):
    if ppn not in death_index.index:
        return None
    death_date = death_index.loc[ppn]['death_date']
    # multiple deaths found
    if type(death_date) == pd.Series:
        # check if they are all the same
        result = np.all(death_date == death_date.iloc[0])
        # if same
        if result:
            return death_date.iloc[0]
        elif drop_duplicates:
            return -1
        # return latest date
        else:
            return death_date.max()
    else:
        return death_date

def get_mh_amb(ppn: str, mh_amb_data: pd.DataFrame, index_start_date: np.timedelta64, date_lookback: np.timedelta64):
    if ppn not in mh_amb_data.index:
        return None
    
    mh_amb_rec = mh_amb_data.loc[[ppn]]
    
    # only include records in the lookback period
    mh_amb_rec = mh_amb_rec[
        (mh_amb_rec['cv_service_contact_date'] <= index_start_date) &
        (mh_amb_rec['cv_service_contact_date'] >= index_start_date - date_lookback)]
    
    if mh_amb_rec.empty:
        return None
    
    treatment_days_num = len(mh_amb_rec)
    return treatment_days_num


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
    parser.add_argument('--lookahead_min', type=int, default=1)
    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()
    if args.config:
        with open(args.config) as fin:
            config = yaml.full_load(fin)

    logging.basicConfig(filename=config['log'], level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)

    diagnosis_cols = ['diagnosis_codeP'] + [f'diagnosis_code{i}' for i in range(1,51)]
    procedure_cols = ['procedure_codeP'] + [f'procedure_code{i}' for i in range(1,51)]

    nrows = None

    if args.all:
        usecols = None
    else:
        usecols = ['episode_start_date', 'episode_end_date', 'ppn', 'episodeset', 
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
                    'apdc_project_recnum'
                    ]
        usecols.extend(diagnosis_cols)
        usecols.extend(procedure_cols)

    # read death records
    deaths = pd.read_feather(config['deaths'], columns=['ppn','DEATH_DATE'])
    deaths.columns = deaths.columns.str.lower()
    deaths['death_date'] = pd.to_datetime(deaths['death_date'], format='%d%b%Y')
    deaths = deaths.set_index('ppn').sort_index()
    
    # cause of death
    cod = pq.read_table(config['cod_urf'], 
                      columns=['ppn', 'ucod_recode', 'death_date']).to_pandas(self_destruct=True)
    cod = cod.set_index('ppn')
    
    # regular expressions for diagnosis and procedures of interest
    diagnosis_procedure_re = build_re_patterns()

    # exclusion stats
    exclusion_criteria = ['follow_up', 'lookback', 'under18', 'death_at_index', 'no_primary_psych', 'los3',
                            'prior_ect', 'any_psych', 'ect_at_index', 'multiple_deaths',
                            'death_error', 'death_at_index', 'ect_date_error']
    exclusion = {e: 0 for e in exclusion_criteria}

    start_date = pd.to_datetime(config['start_date'])
    end_date = pd.to_datetime(config['end_date'])
    date_lookback = pd.offsets.DateOffset(months=config['lookback_months'])
    min_date_lookahead = pd.offsets.DateOffset(months=config['min_followup_months'])
    if config['max_followup_months']:
        max_followup_months = pd.offsets.DateOffset(months=config['max_followup_months'])
    else:
        max_followup_months = None
    ect_counter = 0
    death_counter = 0
    mh_amb_counter = 0
    cod_counter = 0
    PT_GROUPS = 20
    for i in tqdm(range(PT_GROUPS)):
        fname = config['data'].replace('*', str(i))
        mh_fname = config['mh_amb'].replace('*', str(i))
        logging.info(f'Processing {fname}')
        print(fname)
        # read apdc
        df = pd.read_csv(fname, nrows=nrows, usecols=usecols, dtype='str', encoding='ISO-8859-1')
        df = df.astype(str)

        # read mh-amb records
        mh_amb_data = pd.read_parquet(mh_fname, columns=[
            'ppn', 'cv_client_present_status', 'cv_service_contact_date',
            'ca_activity_code', 'ca_principal_serv_category'])
        for col in mh_amb_data.select_dtypes(include=['object']).columns:
            if col != 'ppn':
                mh_amb_data.loc[:, col] = mh_amb_data[col].str.decode('utf-8')
        # only include records where patients were present
        mh_amb_data = mh_amb_data[mh_amb_data['cv_client_present_status']=='1'].reset_index(drop=True)
        mh_amb_data = mh_amb_data.set_index('ppn').sort_index()

        df['age_recode'] = df['age_recode'].astype('float')
        df.loc[:,'age_recode'] = df['age_recode'].fillna(0).astype(float)
        df_sex = pd.get_dummies(df, columns=['SEX'])
        if 'SEX_Indeterminate' not in df_sex.columns:
            df_sex['SEX_Indeterminate'] = 0
        sex_cols = ['SEX_Female', 'SEX_Male', 'SEX_Indeterminate']
        df[sex_cols] = df_sex[sex_cols].astype(int)
        df['episodeset'] = df['episodeset'].astype('int')
        
        df['INVOLUNTARY_DAYS_IN_PSYCH'] = df['INVOLUNTARY_DAYS_IN_PSYCH'].astype('int', errors='ignore')
        df['DAYS_IN_PSYCH_UNIT'] = df['DAYS_IN_PSYCH_UNIT'].astype('int')
        df['EPISODE_LENGTH_OF_STAY'] = df['EPISODE_LENGTH_OF_STAY'].astype('int')    
           
        df['start_date'] = pd.to_datetime(df['episode_start_date'], format='%d%b%Y')
        df['end_date'] = pd.to_datetime(df['episode_end_date'], format='%d%b%Y')
        df['year'] = df['start_date'].apply(lambda x: x.year)

        patient_group = df.groupby('ppn')
        print('Done grouping', flush=True)

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

            # this should never happen because of how raw files were created
            if psych_episodes.empty: 
                exclusion['no_primary_psych'] += 1
                continue

            # keep those with length of stay more than 3 days
            los_mask = psych_episodes['EPISODE_LENGTH_OF_STAY'] >= 3
            psych_episodes = psych_episodes[los_mask]
            
            if psych_episodes.empty: 
                exclusion['los3'] += 1
                continue

            patient_exclusions = set()
            # now safely can check first psych
            matching_episodes = []
            for index, index_admission in psych_episodes.iterrows():  
                if index_admission.age_recode < 18:
                    patient_exclusions.add('under18')
                    continue
                # exclude those with not enough lookback 
                elif index_admission.start_date < start_date+date_lookback:
                    patient_exclusions.add('lookback')
                    continue
                # exclude those with not enough followup data
                elif index_admission.start_date > end_date-min_date_lookahead:
                    patient_exclusions.add('follow_up')
                    continue
    
                # medical history without index admission
                history_no_index = study_group[(study_group.start_date < index_admission.start_date) & (study_group.start_date >= index_admission.start_date - date_lookback)]
                # exclude patients with any severe psychiatric diagnosis during lookback period
                if check_any_psych(history_no_index[diagnosis_cols]):
                    patient_exclusions.add('any_psych')
                    continue
                
                # exclude patients with ECT prior to index psych
                if check_ect(history_no_index[procedure_cols]):
                    patient_exclusions.add('prior_ect')
                    continue

                # episode matches all exclusion criteria
                matching_episodes.append(index_admission.to_dict())
            
            # no episodes that match exclusion criteria 
            if matching_episodes == []:
                for key in patient_exclusions:
                    exclusion[key] += 1
                continue

            # select at random one of the matching episodes to be the index admission
            index_admission = pd.Series(random.choice(matching_episodes))
            index_discharge_date = index_admission.end_date
            # medical history, include index admission
            history = study_group[(study_group.start_date <= index_admission.start_date) & (study_group.start_date >= index_admission.start_date - date_lookback)]
            # medical history without index admission
            history_no_index = study_group[(study_group.start_date < index_admission.start_date) & (study_group.start_date >= index_admission.start_date - date_lookback)]
            
            # count number of prior emergency visits
            emergency_num = len(history_no_index[history_no_index['EMERGENCY_STATUS_RECODE'].str.match('Emergency', na=False, case=False)])
            # count accumulated number of prior days in psychiatric unit
            psych_days_num = history_no_index['DAYS_IN_PSYCH_UNIT'].sum()
            
            # MH-AMB features
            mh_amb_features = get_mh_amb(ppn, mh_amb_data, index_admission.start_date, date_lookback)
            if mh_amb_features == None:
                mh_amb_treatment_days_num = 0
            else:
                mh_amb_treatment_days_num = mh_amb_features
                mh_amb_counter += 1
            
            # get outcomes of interest
            follow_up_features = build_re_patterns_from(config['follow_up_features'])
            cod_features = build_re_patterns_from(config['cod_features'])

            outcomes = {}
            for features in ['follow_up_features', 'cod_features', 'composite_outcome']:
                outcomes_init = {k: 0 for k in config[features].keys()}
                for feature in config[features].keys():
                    outcomes_init[feature+'_date'] = np.nan
                    outcomes_init[feature+'_days'] = np.nan
                outcomes.update(outcomes_init)
                                
            for feature in ['outcome_ect', 'outcome_death', 'outcome_non_suicide']:
                outcomes_init = {feature: 0}
                outcomes_init[feature+'_date'] = np.nan
                if feature != 'outcome_ect':
                    outcomes_init[feature+'_days'] = np.nan
                outcomes.update(outcomes_init)
                        
            # determine end of follow-up, end is defined in yaml
            if max_followup_months:
                individual_end = min(end_date, index_discharge_date + max_followup_months)
            else:
                # all available follow-up
                individual_end = end_date
            
            # this is the patient's followup data starting after index admission
            follow_up = episodes[(study_group.start_date > index_admission.end_date) & 
                                 (study_group.start_date < individual_end)]
            outcomes['end_of_followup_date'] = individual_end
            outcomes['end_of_followup_days'] = (individual_end - index_discharge_date).days
    
            # 0. check if ECT was performed during index admission
            if check_ect(index_admission[procedure_cols]):
                ect_counter += 1
                outcomes['outcome_ect'] = 1
                outcomes['outcome_ect_date'] = index_discharge_date
            else:
                outcomes['outcome_ect'] = 0
                #outcomes['outcome_ect_date'] = outcomes['end_of_followup_date']

            # 1. get death as outcome
            death_date = check_death(ppn, deaths)
            if death_date:
                if death_date == -1:
                    exclusion['multiple_deaths'] += 1
                # if using all available follow-up, will have to revise
                # all code dependent on min_date_lookahead
                elif not max_followup_months or (death_date <= individual_end):
                    # a) all-cause mortality
                    outcomes['outcome_death'] = 1
                    death_counter += 1
                    outcomes['outcome_death_date'] = death_date
                    death_date_days = (death_date - index_discharge_date).days
                    outcomes['outcome_death_days'] = death_date_days
                    # if death is less than admission date, then death was erroneously recorded
                    # if equal, then exclude people who died during index event
                    if death_date_days < 0:
                        exclusion['death_error'] += 1
                        continue
                    # should rarely happen
                    elif death_date_days == 0:
                        exclusion['death_at_index'] += 1
                        continue     
                    outcomes['end_of_followup_date'] = death_date
                    outcomes['end_of_followup_days'] = death_date_days
                    # b) record cause of death - suicide
                    if ppn in cod.index:
                        cause_of_death = get_death_reason(
                            cod.loc[ppn], cod_features,
                            outcomes['end_of_followup_date'], outcomes['end_of_followup_days'])
                        outcomes.update(cause_of_death)
                        cod_counter += 1
                    
                        # c) non-suicide mortality
                        if outcomes['outcome_suicide'] == 0:
                            outcomes['outcome_non_suicide'] = 1
                            outcomes['outcome_non_suicide_date'] = outcomes['end_of_followup_date']
                            outcomes['outcome_non_suicide_days'] = outcomes['end_of_followup_days']
            
            # 2. get individual outcomes coded in yaml
            temp_outcomes = get_outcomes(follow_up, diagnosis_cols, follow_up_features, 
                                    index_discharge_date)
            outcomes.update(temp_outcomes)
            
            # get composite outcomes, defined in yaml
            for label,composite in config['composite_outcome'].items():
                outcomes[label] = int(any([outcomes.get(c, 0) for c in composite]))
                if outcomes[label]:
                    outcome_dates = [outcomes[c+'_date'] for c in composite if type(outcomes[c+'_date']) != float]
                    outcome_days = [outcomes[c+'_days'] for c in composite if type(outcomes[c+'_days']) != float]                   
                    outcomes[label+'_date'] = min(outcome_dates)
                    outcomes[label+'_days'] = min(outcome_days)                        
                else:
                    outcomes[label+'_date'] = np.nan
                    outcomes[label+'_days'] = np.nan

            rec = index_admission.to_dict()
            rec['episodes_prior'] = len(history_no_index)
            rec['emergency_prior'] = emergency_num
            rec['psych_days_num'] = psych_days_num
            rec['mh_amb_treatment_days_num'] = mh_amb_treatment_days_num

            # collect patient history using lookback period
            history_features = get_pt_features(history, diagnosis_procedure_re)
            rec.update(history_features)
            rec['charlson'] = charlson(rec)

            rec.update(dict(sorted(outcomes.items())))
            records.append(rec)

        print('Exclusion criteria')
        for name, value in exclusion.items():
            print(name, value)

        d = pd.DataFrame(records)
        print(d.describe())
        print(d.head())
        print(len(d))
        print(f'ect counter {ect_counter}')
        print(f'death counter {death_counter}')
        print(f'mh-amb counter {mh_amb_counter}')
        print(f'cod counter {cod_counter}')
        
        header = i == 0
        mode = 'a' if i > 0 else 'w'

        d.to_csv(f'survival_sets/outcomes_ect_vs_noect_{start_date.year}_{end_date.year}_back{config["lookback_months"]}_followup{max_followup_months}_rows{nrows}_exclude{config["exclusion"]}.csv', 
                index=False, mode=mode, header=header)
        i += 1

        logging.info('Exclusion criteria')
        for name, value in exclusion.items():
            logging.info(f'{name} {value}')
        logging.info(f'Saved {len(d)} rows')
        logging.info(f'Finished processing {fname}')
    print(f'ect counter {ect_counter}')
    print(f'death counter {death_counter}')
    print(f'mh-amb counter {mh_amb_counter}')
    print(f'cod counter {cod_counter}')

if __name__ == '__main__':
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #ps = pstats.Stats(pr).sort_stats('cumtime')
    #ps.print_stats()
