import pandas as pd

# Raw data   
df = pd.read_csv('survival_sets/raw_ect_stats.csv')
#print(df.shape)
#print(df['ect_num'].describe())
print('Raw Data Stats for patients receiving ECT:\nNumber of ECTs')
print(df[df['ect_num'] != 0]['ect_num'].describe())
print('Average period between ECTs:')
print(df['ect_period'].describe())
print('Age when receiving first ECT:')
print(df['ect_first_age'].describe())
df['ect_num_categories'] = df['ect_num'].apply(lambda x: 3 if x > 3 else x)
print(df['ect_num_categories'].value_counts())

# ECT cohort lookback 12 months
df2 = pd.read_csv(
    'survival_sets/feb_2022_ect_vs_noect_2012_2022_back12_followup-1_none.csv',
    low_memory=False, encoding='ISO-8859-1')
print(f"Death Stats (months):\nDuration of only deaths:\n{df2[df2['observed'] == 2]['duration'].describe()}")
print(f"Duration excluding deaths:\n {df2['duration_ex_death'].describe()}")
print(f"Duration including deaths:\n {df2['duration'].describe()}")
df_2013 = df2[df2['start_date'].str.contains('2013')]
#print(df_2013[df_2013['observed']==0].shape)