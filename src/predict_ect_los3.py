import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import statsmodels.api as sm
import statsmodels.formula.api as smf

from model_utils_ect import get_sa2_quantile, merge_scifa_index, patient_insurance_features, substance_abuse_feature, marital_status_feature, binary_involuntary


def analyze(df, model, fout=None):
    x_cols = [col.strip() for col in model.split('+')]
    X = df[x_cols]
    Y = df[['observed']]

    log_reg = sm.Logit(Y, X)
    res = log_reg.fit(method='bfgs', maxiter=200)
    print(res.summary())
    if fout:
        with open(fout, 'w') as f:
            f.write(res.summary().as_csv())

    # evaluate model fit
    train_auc_scores = []
    train_accuracy = []
    test_auc_scores = []
    test_accuracy = []
    kf = KFold(n_splits=10)
    for train, test in kf.split(df):
        log_reg = sm.Logit(Y.iloc[train], X.iloc[train]).fit(method='bfgs')
        proba_ = log_reg.predict(X.iloc[test])
        test_auc_scores.append(metrics.roc_auc_score(Y.iloc[test], proba_))
        predictions = proba_ > 0.5
        test_accuracy.append(metrics.accuracy_score(Y.iloc[test], predictions))

        proba_ = log_reg.predict(X.iloc[train])
        train_auc_scores.append(metrics.roc_auc_score(Y.iloc[train], proba_))
        predictions = proba_ > 0.5
        train_accuracy.append(metrics.accuracy_score(Y.iloc[train], predictions))

    print(f'Train AUC: {np.mean(train_auc_scores):.2f} ({np.std(train_auc_scores):.2f})')
    print(f'Train acc: {np.mean(train_accuracy):.2f} ({np.std(train_accuracy):.2f})')

    print(f'Test AUC: {np.mean(test_auc_scores):.2f} ({np.std(test_auc_scores):.2f})')
    print(f'Test acc: {np.mean(test_accuracy):.2f} ({np.std(test_accuracy):.2f})')


def experiments(df, output_file):
    base_model = 'age_recode + SEX_Female'
    #analyze(df, model=base_model, fout=f'results/base_{output_file}.csv')
    
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
    df['dep_inv'] = (df['depression'] & df['involuntary_psych']).astype(int)
    clinical_adjustment = clinical_adjustment.replace('other_mood_disorders', 'other_mood_disorders + dep_and')
    clinical_adjustment = clinical_adjustment.replace('involuntary_psych', 'involuntary_psych + dep_inv')

    adjusted_model = base_model + ' + ' + non_clinical_adjustment
    
    # model 3 -> all - clinical and non-clinical
    adjusted_model = f'{base_model} + {clinical_adjustment} + {non_clinical_adjustment}'
    analyze(df, model=adjusted_model, fout=f'results/all_{output_file}.csv')

def main():
    '''
    Script that builds a logistic regression with early ECT as the outcome.
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
    
    df = pd.read_csv(config['binary_survival_data'], low_memory=False, encoding='ISO-8859-1')
            
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
            
    experiments(df, config['binary_output_file'])

if __name__ == '__main__':
    main()
