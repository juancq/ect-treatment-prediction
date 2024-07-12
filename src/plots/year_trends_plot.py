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

    group = df.groupby(['year', 'observed'])

    # number of diagnosis by year/observed
    diag_proportion = group[diagnosis].sum()
    year_total = df.groupby('year')['observed'].count()

    proportion_per_year = diag_proportion.div(year_total, axis=0)
    proportion_per_year = proportion_per_year.reset_index()

    y_max = max(proportion_per_year[diagnosis].max().max(), 0) * 1.05


    for event_val in [0, 1]:
        plt.figure(figsize=(10, 6))
        for diag in diagnosis:
            plt.plot(
                proportion_per_year[(proportion_per_year['observed']==event_val)]['year'],
                proportion_per_year[(proportion_per_year['observed']==event_val)][diag],
                label = f'{diag.title()}'
            )

        title_add = 'Received ECT' if event_val == 1 else 'No ECT'
        plt.title(f'Proportions of diagnoses - {title_add}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(proportion_per_year['year'].unique())
        plt.xlabel('Year')
        plt.ylabel('Percentage')
        plt.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        #plt.ylim(0, y_max)
        plt.tight_layout()
        out_add = 'ect' if event_val == 1 else 'noect'
        plt.savefig(f'diagnosis_yearly_trend_{out_add}.png')




if __name__ == '__main__':
    main()