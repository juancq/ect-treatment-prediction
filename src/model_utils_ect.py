import pandas as pd

def substance_abuse_feature(df):
    df['substance_abuse'] = (df['MDC'] == 'Alcohol/drug use and alcohol/drug induced organic mental disorders').astype(int)
    return df

def get_sa2_quantile(df):
    def decile_to_quantile(decile):
        if decile < 1:
            return 0
        elif decile <= 2:
            return 0
        elif decile <= 4:
            return 1
        elif decile <= 6:
            return 2
        elif decile <= 8:
            return 3
        else:
            return 4
            
    sa2_decile = 'sa2_2016_adv_dis_decile'
    df.loc[:,sa2_decile] = df[sa2_decile].fillna(0)

    df['sa2_2016_q1'] = ((df[sa2_decile] == 1)|(df[sa2_decile] == 2)).astype(int)
    df['sa2_2016_q2'] = ((df[sa2_decile] == 3)|(df[sa2_decile] == 4)).astype(int)
    df['sa2_2016_q3'] = ((df[sa2_decile] == 5)|(df[sa2_decile] == 6)).astype(int)
    df['sa2_2016_q4'] = ((df[sa2_decile] == 7)|(df[sa2_decile] == 8)).astype(int)
    df['sa2_2016_q5'] = ((df[sa2_decile] == 9)|(df[sa2_decile] == 10)).astype(int)
    df['sa2_2016_quantile'] = df[sa2_decile].apply(decile_to_quantile)
    return df


def merge_scifa_index(df, df_scifa):
    #df.loc[:,'SA2_2016_CODE'] = df['SA2_2016_CODE'].str.replace('\..*','', regex=True)
    df.loc[:,'SA2_2016_CODE'] = df['SA2_2016_CODE'].str.replace('^X9.*','-1', regex=True)
    df.loc[:,'SA2_2016_CODE'] = df['SA2_2016_CODE'].str.replace(' ','-1')
    df.loc[:,'SA2_2016_CODE'] = df['SA2_2016_CODE'].fillna(0).astype(int)
    df = pd.merge(df, df_scifa, left_on='SA2_2016_CODE', right_on='sa2_2016_code', how='left')
    return df


def patient_insurance_features(df):
    '''
    See APDC data dictionary for details of how paytment status on separation is coded.
    '''
    #df.loc[:,'public_pt'] = ((df['PAYMENT_STATUS_ON_SEP']<=25)|(df['PAYMENT_STATUS_ON_SEP'].isin([45,46,52,60]))).astype(int)
    #private_pt = (df['PAYMENT_STATUS_ON_SEP']<=39) & (df['PAYMENT_STATUS_ON_SEP']>=30)
    #other_pt = df['PAYMENT_STATUS_ON_SEP'].isin([40, 41, 42, 50, 51, 55])
    df.loc[:,'public_pt'] = (
        (df['PAYMENT_STATUS_ON_SEP'].str.contains('Public'))
        |(df['PAYMENT_STATUS_ON_SEP'].str.contains('Unqualified'))
        |(df['PAYMENT_STATUS_ON_SEP'].str.contains('Residential Aged Care - Other'))
        |(df['PAYMENT_STATUS_ON_SEP'].str.contains('Other Ineligible'))
    ).astype(int)
    private_pt = (df['PAYMENT_STATUS_ON_SEP'].str.contains('Private'))
    other_pt = (
        (df['PAYMENT_STATUS_ON_SEP'].str.contains('Compensable'))
        |(df['PAYMENT_STATUS_ON_SEP'].str.contains('Veterans'))
        |(df['PAYMENT_STATUS_ON_SEP'].str.contains('Defence'))
    )
    df.loc[:,'private_pt'] = (private_pt | other_pt).astype(int)
    return df

def marital_status_feature(df):
    df['marital_status'] = (df['MARITAL_STATUS'] == 'Married (including de facto)').astype(int)
    return df

def binary_involuntary(df):
    df['involuntary_psych'] = df['INVOLUNTARY_DAYS_IN_PSYCH'].apply(lambda c: 1 if c>=1 else 0)
    return df

def binary_diagnoses(df):
    psych_diagnoses = ['depression', 'bipolar_disorder', 'schizo','schizo_other',
    'mania', 'other_mood_disorders']
    
    charlson_diagnoses = ['charlson_myocardial_infarction',
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
    'charlson_aids_hiv']
    
    for diagnosis in psych_diagnoses + charlson_diagnoses:
        df[diagnosis] = df[diagnosis].apply(lambda c: 1 if c>=1 else 0)
    return df

def get_age_group(df):
    def simplify_age_group(age):
        if age == '15 - 19 years' or age == '20 - 24 years' or age == '25 - 29 years':
            return '18 - 29 years'
        elif age == '30 - 34 years' or age == '35 - 39 years' or age == '40 - 44 years':
            return '30 - 44 years'
        elif age == '45 - 49 years' or age == '50 - 54 years' or age == '55 - 59 years':
            return '45 - 59 years'
        elif age == '60 - 64 years' or age == '65 - 69 years' or age == '70 - 74 years':
            return '60 - 74 years'
        else:
            return '75+ years'
        
    df['age_grouping_recode'] = df['age_grouping_recode'].apply(simplify_age_group)
    return df