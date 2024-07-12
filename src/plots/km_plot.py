import argparse
import yaml
import pandas as pd
from lifelines.plotting import add_at_risk_counts
from lifelines import NelsonAalenFitter
from matplotlib import pyplot as plt


def main():
    '''
    Script for generating KM plot (Figure 1).
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    
    args = parser.parse_args()
    if args.config:
        with open(args.config) as fin:
            config = yaml.safe_load(fin)
    args = parser.parse_args()
        
    df = pd.read_csv(config['survival_data'], usecols=['SEX_Female', 'year', 'observed', 'duration'],low_memory=False,encoding='ISO-8859-1')

    #df = df[df['year'] > 2012]
    # remove people who died on index admission
    #dead_mask = (df.observed==0) & (df.duration < 0.01)
    #df = df[~dead_mask].copy()

    print('Median [IQR] follow-up:')
    print(f'{(df.duration/12).quantile([0.25, 0.5, 0.75])}')

    df.duration = df.duration / 12

    # for plotting death

    death = False

    # for 36 months
    if '36' in config['survival_data']:
        ticks = range(0,8)
    else:
    # for 12 months
        ticks = range(0,10)

    if death:
        df.observed.replace(1, 0, inplace=True)
        df.observed.replace(2, 1, inplace=True)
        title = 'Death'
    else:
        df.observed.replace(2, 0, inplace=True)
        title = 'Receiving ECT'

    at_risk_counts = True
    def plot_na(df, label=''):
        ax = plt.subplot(111)
        naf = NelsonAalenFitter(nelson_aalen_smoothing=False)
        naf.fit(df['duration'], event_observed=df['observed'], label='')
        naf.plot_cumulative_hazard(ax=ax, legend=False)#, at_risk_counts=at_risk_counts)
        plt.title(f'Cumulative Hazard Rate of {title}')
        #plt.title('Cumulative Hazard Rate of Receiving ECT')
        add_at_risk_counts(naf, xticks=ticks)
        #ax.set_xlim((0,120))
        #ax.set_xticks(list(range(0,120+12,12)))
        #ax.set_xticklabels(list(range(11)))
        ax.set_xticks(list(ticks))
        ax.grid(which='major', color='silver', linewidth=1.0, alpha=0.3)
        ax.set_xlabel('Years')
        folder = f''
        plt.savefig(f'{folder}na_plot_ect_vs_noect_{label}.png', bbox_inches='tight', dpi=300)
    
    plot_na(df, label=config['output_file'])


if __name__ == '__main__':
    main()
