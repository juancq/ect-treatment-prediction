import argparse
import glob
import re
import yaml
import pandas as pd
import numpy as np

from survival_utils_ect import filter_transfers
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


any_psych_re = re.compile('F(?:2[0-9]|3[0-9])')
def check_any_psych(df):
    df_flat = df.astype(str).values.flatten()
    return np.array([bool(any_psych_re.search(i)) for i in df_flat]).any() 
                

ECT_CODES = '14224-0[0-6]'
ect_re = re.compile(ECT_CODES)
def check_ect(df):
    df_flat = df.astype(str).values.flatten()
    return np.array([bool(ect_re.search(i)) for i in df_flat]).any() 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_data', metavar='episode_data', type=str, nargs='+',
                        help='patient chunks of episode data from APDC')
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--dropna', action='store_true')
    parser.add_argument('--nrows', type=int, default=None)
    parser.add_argument('--lookahead_min', type=int, default=1)
    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()
    if args.config:
        with open(args.config) as fin:
            config = yaml.full_load(fin)

    diagnosis_cols = ['diagnosis_codeP'] + [f'diagnosis_code{i}' for i in range(1,51)]
    procedure_cols = ['procedure_codeP'] + [f'procedure_code{i}' for i in range(1,51)]

    nrows = args.nrows

    if args.all:
        usecols = None
    else:
        usecols = ['episode_start_date', 'ppn', 'episodeset', 
                    'age_recode',
                    ]
        usecols.extend(diagnosis_cols)
        usecols.extend(procedure_cols)

    # read death records
    deaths = pd.read_feather(config['deaths'], columns=['ppn','DEATH_DATE'])
    deaths.columns = deaths.columns.str.lower()
    deaths['death_date'] = pd.to_datetime(deaths['death_date'], format='%d%b%Y')
    deaths = deaths.set_index('ppn')

    end_date = pd.to_datetime(config['end_date'])
    i = 0
    any_psych_drop = 0
    ect_drop = 0
    ect_counter = 0
    for fname in tqdm(sorted(glob.glob(config['data']))):
        print(fname)
        df = pd.read_csv(fname, nrows=nrows, usecols=usecols, dtype='str', encoding='ISO-8859-1')
        df['age_recode'] = df['age_recode'].astype('float')
        df.loc[:,'age_recode'] = df['age_recode'].fillna(0).astype(float)
        df['episodeset'] = df['episodeset'].astype('int')
                   
        df['start_date'] = pd.to_datetime(df['episode_start_date'], format='%d%b%Y')
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
            index_admission = study_group.iloc[0]

            # find ECT procedures in follow-up
            ect = study_group[procedure_cols].apply(lambda c: c.str.contains(ECT_CODES, na=False)).any(axis=1)
            # count number of ECT procedures in total
            ect_num = study_group[procedure_cols].apply(lambda c: c.str.contains(ECT_CODES)).sum().sum()

            # ECT was never performed, right-censored
            if not ect.any():
                observed = 0

                # check for death, censor on death by setting observed=2
                death_date = check_death(ppn, deaths)

                # this should only happen if we decide to drop all patients with erroneous death records
                # i.e. calling check_death with drop_duplicates=True
                if death_date == -1:
                    duration = (end_date - index_admission.start_date).days / 365 * 12
                    # skip this patient
                    #continue
                # death
                elif death_date is not None:
                    # checking to be pedantic
                    if death_date < end_date:
                        duration = (death_date - index_admission.start_date).days / 365 * 12
                        # set 2 as death event
                        observed = 2
                else:
                    duration = (end_date - index_admission.start_date).days / 365 * 12

                ect_date = None
                ect_period = None
                ect_first_age = None
            else:
                ect_counter += 1
                # ECT was performed during follow-up
                ect_episode = study_group[ect].iloc[0]
                # this contains days of follow-up when ECT occurred
                diff = (ect_episode.start_date - index_admission.start_date)
                # period between ECTs
                ect_period = study_group[ect].start_date.diff().dt.days.mean()
                # age of first ECT recorded
                ect_first_age = ect_episode['age_recode']
                # sometimes this happens
                # it seems to be that the records are out of order, so dates are not sorted, but
                # technically these are coming from strata
                if diff.days < 0:
                    print('debugging ', ect_episode.start_date, index_admission.start_date)
                    print(study_group['start_date'])
                    continue

                observed = 1
                duration = diff.days/365 * 12

                ect_date = ect_episode.start_date

            rec = index_admission.to_dict()
            rec['observed'] = observed
            rec['duration'] = duration
            rec['ect_date'] = ect_date
            rec['ect_num'] = ect_num
            rec['ect_period'] = ect_period
            rec['ect_first_age'] = ect_first_age

            records.append(rec)

        print(f'drops psych {any_psych_drop} ect {ect_drop}')
        d = pd.DataFrame(records)
        print(d['ect_num'].describe())
        print(d['ect_period'].describe())
        print(d['ect_first_age'].describe())
        print(d['duration'].describe())
        print(d['duration'][d['duration']<0])
        print(d.head())
        print(len(d))
        print(f'ect counter {ect_counter}')
        
        header = i == 0
        mode = 'a' if i > 0 else 'w'

        d.to_csv('survival_sets/raw_ect_stats.csv', 
                index=False, mode=mode, header=header)
        i += 1

    print(f'ect counter {ect_counter}')
    print(d['ect_num'].describe())
    print(d['ect_period'].describe())
    print(d['ect_first_age'].describe())
    print(d['duration'].describe())


if __name__ == '__main__':
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #ps = pstats.Stats(pr).sort_stats('cumtime')
    #ps.print_stats()
