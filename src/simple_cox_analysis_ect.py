import argparse
import yaml
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold

from model_utils_ect import get_sa2_quantile, merge_scifa_index, patient_insurance_features, substance_abuse_feature, marital_status_feature, binary_involuntary
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data['Variable'] = df.columns
    vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    print(vif_data)


def analyze(df, model, fout=None):
    cph = CoxPHFitter()
    cph.fit(df, duration_col='duration', event_col='observed', formula=model)
    cph.print_summary(decimals=3)
    if fout:
        cph.summary.to_csv(fout)

    # evaluate model fit
    scores = []
    kf = KFold(n_splits=10)
    for train, test in kf.split(df):
        cph.fit(df.iloc[train], duration_col='duration', event_col='observed', formula=model)
        test_score = cph.score(df.iloc[test], scoring_method='concordance_index')
        scores.append(test_score)

    test_score = np.mean(scores)
    test_std = np.std(scores) 
    print(f'Test: {test_score:.2f} ({test_std:.2f})')


def experiments(df, output_file):
    base_model = 'age_recode + SEX_Female'
    analyze(df, model=base_model, fout=f'results/base_{output_file}.csv')

    # model 1 -> clinical factors
    clinical_adjustment = 'bipolar_disorder + schizo + schizo_other'
    clinical_adjustment += '+ mania + other_mood_disorders'
    clinical_adjustment += '+ involuntary_psych'
    clinical_adjustment += '+ psych_days_num'
    clinical_adjustment += '+ mh_amb_treatment_days_num'
    clinical_adjustment += '+ emergency_prior +episodes_prior'
    clinical_adjustment += '+substance_abuse'
    clinical_adjustment += '+ charlson_myocardial_infarction'
    clinical_adjustment += '+ charlson_congestive_heart_failure'
    clinical_adjustment += '+ charlson_peripheral_vascular_disease'
    clinical_adjustment += '+ charlson_renal_disease'
    clinical_adjustment += '+ charlson_moderate_severe_liver_disease'
    clinical_adjustment += '+ charlson_mild_liver_disease'
    clinical_adjustment += '+ charlson_peptic_ulcer_disease'
    clinical_adjustment += '+ charlson_rheumatic_disease'
    clinical_adjustment += '+ charlson_cerebrovascular_disease'
    clinical_adjustment += '+ charlson_dementia'
    clinical_adjustment += '+ charlson_chronic_pulmonary_disease'
    clinical_adjustment += '+ charlson_diabetes_uncomplicated'
    clinical_adjustment += '+ charlson_diabetes_complicated'
    clinical_adjustment += '+ charlson_hemiplegia_paraplegia'
    clinical_adjustment += '+ charlson_malignancy'

    adjustment_cols = [i.strip() for i in clinical_adjustment.split('+')]
    df.loc[:,adjustment_cols] = df[adjustment_cols].astype(bool).astype(int)

    non_clinical_adjustment = ' year + private_pt'
    non_clinical_adjustment += '+ sa2_2016_quantile'
    non_clinical_adjustment += '+ marital_status'

    psych_diagnoses = ['bipolar_disorder', 'schizo','schizo_other',
    'mania', 'other_mood_disorders']
    mask = (df['depression'] & (df[psych_diagnoses].sum(axis=1)).astype(bool))
    df.loc[mask,psych_diagnoses] = 0
    df['dep_and'] = mask.astype(int)
    mask_inv_psych = (df['depression'] & df['involuntary_psych'])
    df.loc[mask_inv_psych,'involuntary_psych'] = 0
    df['dep_inv'] = mask_inv_psych.astype(int)

    clinical_adjustment = clinical_adjustment.replace('other_mood_disorders', 'other_mood_disorders + dep_and')
    clinical_adjustment = clinical_adjustment.replace('involuntary_psych', 'involuntary_psych + dep_inv')

    # model 2 -> all - clinical and non-clinical
    adjusted_model = f'{base_model} + {clinical_adjustment} + {non_clinical_adjustment} + dep_and + dep_inv'
    analyze(df, model=adjusted_model, fout=f'results/all_{output_file}.csv')


def main():
    '''
    Fits four Cox models using (1) sex and age, (2) clinical factors, 
    (3) sociodemographic factors, and (4) all variables.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, help='output file to save summary results')
    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()
    if args.config:
        with open(args.config) as fin:
            config = yaml.safe_load(fin)
    args = parser.parse_args()

    np.random.seed(123)

    df = pd.read_csv(config['survival_data'], low_memory=False, encoding='ISO-8859-1')
    df.observed.replace(2, 0, inplace=True)

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

    experiments(df, config['output_file'])


if __name__ == '__main__':
    main()
