import argparse
import yaml
import pandas as pd
import numpy as np
import sys
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

from tableone import TableOne
from model_utils_ect import get_sa2_quantile, merge_scifa_index, patient_insurance_features, substance_abuse_feature, marital_status_feature, binary_involuntary, binary_diagnoses, get_age_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()
    if args.config:
        with open(args.config) as fin:
            config = yaml.safe_load(fin)
    args = parser.parse_args()

    np.random.seed(123)

    df = pd.read_csv(config['survival_data'], low_memory=False, encoding='ISO-8859-1')
    df.observed.replace(2, 0, inplace=True)

    diagnosis =['depression','bipolar_disorder','mania','schizo', 'schizo_other', 'other_mood_disorders']
    df.loc[:,diagnosis] = (df[diagnosis] > 0).astype(int)

    years = sorted(df.year.unique())
    for yr in years:
        df[str(yr)] = (df.year==yr).astype(int)

    categorical = list(map(str,years))
    usecols = ['observed'] + categorical
    table1 = TableOne(df, columns=usecols, 
                groupby='observed',
                categorical=categorical,
                pval=True,
                htest_name=True)

    print(table1.tabulate(tablefmt='fancy_grid'))
    table1.to_csv(f"results/year_trends_table.csv")


if __name__ == '__main__':
    main()
