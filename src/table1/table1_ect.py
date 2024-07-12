import argparse
import yaml
import pandas as pd
import numpy as np
import sys

from tableone import TableOne
from model_utils_ect import get_sa2_quantile, merge_scifa_index, patient_insurance_features, substance_abuse_feature, marital_status_feature, binary_involuntary, binary_diagnoses, get_age_group


def main():
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()
    if args.config:
        with open(args.config) as fin:
            config = yaml.safe_load(fin)
    args = parser.parse_args()

    np.random.seed(123)

    df = pd.read_csv(config['survival_data'], low_memory=False, encoding='ISO-8859-1')

    # generating columns that were used in the model
    # get socioeconomic status based on residence area
    df_scifa = pd.read_csv(config['scifa'], dtype='Int64', encoding='ISO-8859-1')
    # merge with patient features
    df = merge_scifa_index(df, df_scifa)
    # stratify socioeconomic index into quantiles
    df = get_sa2_quantile(df)

    # get private insurance features
    df = patient_insurance_features(df)

    # get marital status feature
    df = marital_status_feature(df)
       
    # get substance abuse feature
    df = substance_abuse_feature(df)

    # get involuntary psych feature
    df = binary_involuntary(df)

    # get psych and charlson diagnoses features
    df = binary_diagnoses(df)

    #get recoded age group features
    df = get_age_group(df)

    psych_diagnoses = ['bipolar_disorder', 'schizo','schizo_other', 'mania', 'other_mood_disorders']
    new_psych = []
    for p in psych_diagnoses:
        for dep_var in ['depression', 'private_pt', 'involuntary_psych']:
            df[f'{p}*{dep_var}'] = df[p] * df[f'{dep_var}']
            new_psych.append(f'{p}*{dep_var}')
    
    for dep_var in ['private_pt', 'involuntary_psych']:
        df[f'depression*{dep_var}'] = df['depression'] * df[f'{dep_var}']
        new_psych.append(f'depression*{dep_var}')

    psych_diagnoses.extend(new_psych)
    return

    '''
    df['observed'] = df['observed'].replace(2, 0)
    print(df[(df['observed']==1) & (df['schizo'] == 1)]['duration'].mean())
    print(df[(df['observed']==1) & (df['schizo'] == 0)]['duration'].mean())
    print(df[(df['observed']==1) & (df['schizo_other'] == 1)]['duration'].mean())
    print(df[(df['observed']==1) & (df['schizo_other'] == 0)]['duration'].mean())
    print(df[(df['observed']==1) & (df['depression'] == 1)]['duration'].mean())
    print(df[(df['observed']==1) & (df['depression'] == 0)]['duration'].mean())
    print(df[(df['observed']==1) & (df['private_pt'] == 1)]['duration'].mean())
    print(df[(df['observed']==1) & (df['private_pt'] == 0)]['duration'].mean())
    return
    '''

    df['depression_new'] = df['depression'] & ~df['schizo']
    df['schizo_new'] = ~df['depression'] & df['schizo']

    usecols = ['observed', 'age_recode', 'age_grouping_recode', 'SEX_Female',
               'SEX_Male', 'SEX_Indeterminate',
               'charlson', 'charlson_myocardial_infarction',
               'charlson_congestive_heart_failure',
               'charlson_peripheral_vascular_disease',
               'charlson_renal_disease',
               'charlson_moderate_severe_liver_disease',
               'charlson_mild_liver_disease',
               'charlson_peptic_ulcer_disease',
               'charlson_rheumatic_disease',
               'charlson_cerebrovascular_disease',
               'charlson_dementia',
               'charlson_chronic_pulmonary_disease',
               'charlson_diabetes_uncomplicated',
               'charlson_diabetes_complicated',
               'charlson_hemiplegia_paraplegia',
               'charlson_malignancy',
               'charlson_metastatic_solid_tumour',
               'charlson_aids_hiv',
               'substance_abuse',
               'DAYS_IN_PSYCH_UNIT', 'involuntary_psych',
               'emergency_prior', 'EPISODE_LENGTH_OF_STAY',
               'episodes_prior', 'ect_num', 'duration',
               'mh_amb_treatment_days_num',
               'private_pt', 'sa2_2016_quantile',
               'sa2_2016_q1', 'sa2_2016_q2', 'sa2_2016_q3', 'sa2_2016_q4', 'sa2_2016_q5',
               'marital_status',
               ]

    psych_diagnoses.extend(['depression_new', 'schizo_new'])
    usecols.extend(psych_diagnoses)

    categorical = ['age_grouping_recode', 'SEX_Female', 'SEX_Male', 'SEX_Indeterminate',
                   'charlson_myocardial_infarction',
                   'charlson_congestive_heart_failure',
                   'charlson_peripheral_vascular_disease',
                   'charlson_renal_disease',
                   'charlson_moderate_severe_liver_disease',
                   'charlson_mild_liver_disease',
                   'charlson_peptic_ulcer_disease',
                   'charlson_rheumatic_disease',
                   'charlson_cerebrovascular_disease',
                   'charlson_dementia',
                   'charlson_chronic_pulmonary_disease',
                   'charlson_diabetes_uncomplicated',
                   'charlson_diabetes_complicated',
                   'charlson_hemiplegia_paraplegia',
                   'charlson_malignancy',
                   'charlson_metastatic_solid_tumour',
                   'charlson_aids_hiv',
                   'substance_abuse', 'involuntary_psych', 'private_pt',
                   'sa2_2016_quantile',
                   'sa2_2016_q1', 'sa2_2016_q2', 'sa2_2016_q3', 'sa2_2016_q4', 'sa2_2016_q5',
                   'marital_status'
                   ] + psych_diagnoses

    groupby = 'observed'
    nonnormal = [
                'age_recode',
                'charlson',
               'DAYS_IN_PSYCH_UNIT', 
               'EPISODE_LENGTH_OF_STAY',
               'emergency_prior', 
               'episodes_prior', 
               'duration',
               'mh_amb_treatment_days_num',]

    #df['observed'] = df['observed'].replace(2, 0)

    contingency_table = pd.crosstab(df['depression'], df['schizo'])
    print(contingency_table)

    df['observed'] = df['observed'].replace(2, 0)

    contingency_table = pd.crosstab(df['depression'], df['schizo'], df['observed'], aggfunc='sum', margins=False, normalize='index')
    print(contingency_table)

    contingency_table = pd.crosstab(df['depression'], df['schizo'], values=df['observed'], aggfunc='sum')
    print(contingency_table)

    contingency_table = pd.crosstab(df['depression']*df['observed'], df['schizo']*df['observed'], margins=True)#, normalize='index')
    print(contingency_table)
    #return

    #table1 = TableOne(df[df['SEX_Female']==1], columns=usecols, categorical=categorical, 
    #            groupby=groupby, nonnormal=nonnormal,
    #            htest_name=True,
    #            pval=True)
    #print(table1.tabulate(tablefmt='fancy_grid'))
    #table1 = TableOne(df[df['SEX_Female']!=1], columns=usecols, categorical=categorical, 
    #            groupby=groupby, nonnormal=nonnormal,
    #            htest_name=True,
    #            pval=True)
    #print(table1.tabulate(tablefmt='fancy_grid'))

    table1 = TableOne(df,columns=usecols, categorical=categorical, 
                groupby=groupby, nonnormal=nonnormal,
                htest_name=True,
                pval=True)
    print(table1.tabulate(tablefmt='fancy_grid'))

    table1.to_csv(f"results/intersection_depression.csv")


if __name__ == "__main__":
    main()
