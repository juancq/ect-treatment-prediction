import re
import numpy as np
import pandas as pd


def condense_codes(df, index, label):
    # concatenate all procedures into procedure_codeP field
    # combine all diagnosis, keep unique codes, count them, and put them all in diagnosis fields 1-50
    # use the rest of the information from the very first record
    _i = index.index[0]
    cols = [f'{label}_code{i}' for i in range(1,51)]
    codes = list(set(filter(pd.notnull, df[cols].fillna('').values.flatten().tolist())))
    num_codes = len(codes)
    if num_codes > 50:
        codes = codes[:49] + [','.join(codes[:50])]
        num_codes = 50

    cols = [f'{label}_code{i}' for i in range(1,num_codes+1)]
    index.loc[_i,cols] = codes

    primary_col = f'{label}_codeP'
    primary_code = list(set(filter(pd.notnull, df[primary_col].tolist())))
    index.loc[_i,primary_col] = ','.join(primary_code)


def filter_transfers(df):
    if len(df) == 1:
        return df
    else:
        # this is a transfer

        # add up all time variables that make up single hospital stay
        time_cols = ['DAYS_IN_PSYCH_UNIT', 'EPISODE_LEAVE_DAYS_TOTAL', 
                    'INVOLUNTARY_DAYS_IN_PSYCH', 'EPISODE_LENGTH_OF_STAY']
        #df[time_cols] = df[time_cols].sum()

        index = df.iloc[[0]]
        _i = index.index[0]
        index.loc[_i,time_cols] = df[time_cols].sum()

        condense_codes(df, index, 'procedure')
        condense_codes(df, index, 'diagnosis')
        return index


def pseudo_lambda(df, regex):
    return np.array([bool(regex.search(i)) for i in df])


def build_re_patterns():
    psych = [
        ['depression', 'F3[23]'],
        ['bipolar_disorder', 'F31'],
        ['mania', 'F30'],
        # per Colleen
        # schizophrenia and schizoactive
        ['schizo', 'F2[05]'],
        ['schizo_other', 'F2[1-489]'],
        ['other_mood_disorders', 'F3[4-9]'],
        #['depressive_episode', 'F32'],
        #['depressive_disorder', 'F33'],
        #['schizotypal', 'F21'],
        #['delusional', 'F22'],
        #['psychotic', 'F23'],
        #['induced_delusional', 'F24'],
        #['schizoaffective', 'F25'],
        #['nonorganic_psychotic', 'F28'],
        #['unspecified_nonorganic_psychotic', 'F29'],
    ]

        #['alcohol', 'K70,T52,K86.0,E52']

    # charlson
    charlson = [
        ['charlson_myocardial_infarction', 'I2[12],I25.2'],
        ['charlson_congestive_heart_failure', 'I09.9,I11.0,I13.[02],I25.5,I42.[05-9],I43,I50,P29.0'],
        ['charlson_peripheral_vascular_disease', 'I7[01],I73.[189],I77.1,I79.[02],K55.[189],Z95.[89]'],

        ['charlson_renal_disease', 'I12.0,I13.1,N03.[2-7],N1[89],N25.0,Z49.[0-2],Z94.0,Z99.2'],
        ['charlson_moderate_severe_liver_disease', 'I85.[09],I86.4,I98.2,K70.4,K71.1,K72.[19],K76.[5-7]'],
        ['charlson_mild_liver_disease', 'B18,K70.[0-39],K71.[3-57],K7[34],K76.[02-489],Z94.4'],

        ['charlson_peptic_ulcer_disease', 'K2[5-8].'],
        ['charlson_rheumatic_disease', 'M0[56].,M31.5,M3[2-4].,M35.[13],M36'],
        ['charlson_cerebrovascular_disease', 'G4[56].,H34.0,I6[0-9].'],
        ['charlson_dementia', 'F0[0-3].,F05.1,G30.,G31.1'],
        ['charlson_chronic_pulmonary_disease', 'I27.[89],J4[0-7].,J6[0-7].,J68.4,J70.[13]'],
        ['charlson_diabetes_uncomplicated', 'E10.[01689],E11.[01689],E12.[01689],E13.[01689],E14.[1689]'],
        ['charlson_diabetes_complicated', 'E10.[2-57],E11.[2-57],E12.[2-57],E13.[2-57],E14.[2-57]'],
        ['charlson_hemiplegia_paraplegia', 'G04.1,G11.4,G80.[12],G81.,G82.,G83.[0-49]'],
        ['charlson_malignancy', 'C[0-2][0-6].,C3[0-4].,C3[7-9].,C4[0135-9].,C5[0-8].,C6[0-9].,C7[0-6].,C8[1-58].,C9[0-7].'],
        ['charlson_metastatic_solid_tumour', 'C[7-9].,C80.'],
        ['charlson_aids_hiv', 'B2[0-24].']
    ]

    #procedures = [
    #    ['ect', '14224-0[0-6]'],
    #]

    #feature_sets = psych + charlson + procedures
    feature_sets = psych + charlson

    feature_extraction_re = []
    for name,codes in feature_sets:
        split_codes = [c.replace('.','\.') for c in codes.split(',')]
        regex = '|'.join([f'(?:{c})' for c in split_codes])
        r = re.compile(regex)
        feature_extraction_re.append((name,r))

    return feature_extraction_re


def get_pt_features(df, feature_re):
    features = {}

    diagnosis_cols = ['diagnosis_codeP'] + [f'diagnosis_code{i}' for i in range(1,51)]
    proc_cols = ['procedure_codeP'] + [f'procedure_code{i}' for i in range(1,51)]

    df_temp = df[diagnosis_cols + proc_cols]
    df_flat = df_temp.values.flatten()

    for feature_name, regex in feature_re:
        features[feature_name] = pseudo_lambda(df_flat, regex).sum()

    return features


def charlson(patient):
    score = 0
    
    if patient['age_recode'] >= 50 and patient['age_recode'] <60:
        score += 1
    elif patient['age_recode'] >=60 and patient['age_recode'] <70:
        score += 2
    elif patient['age_recode'] >=70 and patient['age_recode'] <80:
        score += 3
    elif patient['age_recode'] >= 80:
        score += 4

    if patient['charlson_myocardial_infarction']:
        score += 1

    if patient['charlson_congestive_heart_failure']:
        score += 1
        
    if patient['charlson_peripheral_vascular_disease']:
        score += 1
    
    if patient['charlson_cerebrovascular_disease']:
        score += 1
       
    if patient['charlson_dementia']:
        score += 1
    
    if patient['charlson_chronic_pulmonary_disease']:
        score += 1
        
    if patient['charlson_rheumatic_disease']:
        score += 1
    
    if patient['charlson_peptic_ulcer_disease']:
        score += 1
        
    if patient['charlson_mild_liver_disease']:
        score += 1
        
    if patient['charlson_moderate_severe_liver_disease']:
        score += 3
    
    if patient['charlson_diabetes_uncomplicated']:
        score += 1

    if patient['charlson_diabetes_complicated']:
        score += 2

    if patient['charlson_hemiplegia_paraplegia']:
        score += 2

    if patient['charlson_renal_disease']:
        score += 2

    if patient['charlson_malignancy']:
        score += 2
        
    if patient['charlson_metastatic_solid_tumour']:
        score += 6
        
    if patient['charlson_aids_hiv']:
        score += 6

    return score

def get_outcomes(df, diagnosis_cols, outcome_features, index_date):
    features = dict.fromkeys(outcome_features.keys(), 0)
    for outcome,outcome_code in outcome_features.items():
        mask = df[diagnosis_cols].apply(lambda c: c.str.contains(outcome_code, regex=True, na=False)).any(axis=1)
        features[outcome] = mask.any().astype(int)
        if features[outcome]:
            first_outcome_date = df[mask]['start_date'].min()
            features[outcome + '_date'] = first_outcome_date
            features[outcome + '_days'] = (first_outcome_date - index_date).days
    return features


def get_death_reason(record, outcome_features, followup_date, followup_days):
    features = {}
    if isinstance(record.ucod_recode, str):
        for outcome,outcome_code in outcome_features.items():
            match = bool(re.search(outcome_code, record.ucod_recode))
            if match:
                features[outcome] = int(match)
                features[outcome + '_date'] = followup_date
                features[outcome + '_days'] = followup_days
    return features

def build_re_patterns_from(feature_sets, pandas=False):
        
    feature_extraction_re = {}
    for name,codes in feature_sets.items():
        split_codes = [c.strip().replace('.','\.') for c in codes.split(',')]
        regex = '|'.join([f'(?:{c})' for c in split_codes])
        if not pandas:
            regex = re.compile(regex)
        feature_extraction_re[name] = regex

    return feature_extraction_re
