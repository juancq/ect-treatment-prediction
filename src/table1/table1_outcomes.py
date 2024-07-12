import argparse
import yaml
import pandas as pd
import numpy as np

from tableone import TableOne
from model_utils_ect import get_sa2_quantile, merge_scifa_index, patient_insurance_features, substance_abuse_feature, marital_status_feature, binary_involuntary, binary_diagnoses, get_age_group

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)

args = parser.parse_args()
if args.config:
    with open(args.config) as fin:
        config = yaml.safe_load(fin)
args = parser.parse_args()

np.random.seed(123)

df = pd.read_csv(config['outcomes_data'], low_memory=False, encoding='ISO-8859-1')
df = df[df['year'] > 2012]
df.loc[:,'age_recode'] = df['age_recode'].fillna(0).astype(float)

# remove people who died on index admission
#dead_mask = (df.observed==0) & (df.duration < 0.01)
#df = df[~dead_mask].copy()

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

usecols = ['outcome_psych_hospitalization', 'outcome_psych_hospitalization_days',
           'outcome_self_harm', 'outcome_self_harm_days',
           'outcome_death', 'outcome_death_days',
           'outcome_suicide', 'outcome_suicide_days',
           'outcome_non_suicide', 'outcome_non_suicide_days',
           'suicide_and_readmission', 'suicide_and_readmission_days',
           'age_recode', 'age_grouping_recode', 'SEX_Female',
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
           'depression', 'bipolar_disorder', 'schizophrenia',
           'mania', 'other_mood_disorders',
           'substance_abuse',
           'DAYS_IN_PSYCH_UNIT', 'involuntary_psych',
           'emergency_prior', 'psych_days_num', 'EPISODE_LENGTH_OF_STAY',
           'episodes_prior',
           'mh_amb_treatment_days_num',
           'year', 'private_pt', 'sa2_2016_quantile',
           'sa2_2016_q1', 'sa2_2016_q2', 'sa2_2016_q3', 'sa2_2016_q4', 'sa2_2016_q5',
           'marital_status'
           ]

categorical = ['outcome_psych_hospitalization', 'outcome_self_harm',
               'outcome_death', 'outcome_suicide', 'outcome_non_suicide',
               'suicide_and_readmission',
               'age_grouping_recode', 'SEX_Female', 'SEX_Male', 'SEX_Indeterminate',
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
               'depression', 'bipolar_disorder', 'schizophrenia',
               'mania', 'other_mood_disorders',
               'substance_abuse', 'involuntary_psych', 'private_pt',
               'sa2_2016_quantile',
               'sa2_2016_q1', 'sa2_2016_q2', 'sa2_2016_q3', 'sa2_2016_q4', 'sa2_2016_q5',
               'marital_status'
               ]

groupby = 'outcome_ect'

table1 = TableOne(df, columns=usecols, categorical=categorical, groupby=groupby, pval=True)

print(table1.tabulate(tablefmt='fancy_grid'))

output = config['outcomes_output_file']
table1.to_csv(f"ect_{output}/table1_{output}.csv")