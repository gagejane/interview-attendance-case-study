import pandas as pd

df = pd.read_csv('data/interview.csv')
pd.options.display.max_columns = 28


millie_cols = ['Candidate Current Location',
       'Candidate Job Location', 'Interview Venue',
       'Candidate Native location', 'Observed Attendance']


df = df[millie_cols]

df2 = df.copy()
cols = df2.columns.tolist()
cols = [col.replace(' ', '_').lower() for col in cols]
df2.columns = cols

for col in df2.columns:
    for index in range(len(df2[col])):
        if type(df2.loc[index, col]) == str:
            df2.loc[index, col] = df2.loc[index, col].replace('-', ' ')
            df2.loc[index, col] = df2.loc[index, col].lower().strip()
            df2.loc[index, col] = df2.loc[index, col].replace('delhi /ncr', 'delhi')

df2['travel'] = df2['interview_venue'] != df2['candidate_current_location']
df2['hometown'] = df2['candidate_job_location'] == df2['candidate_native_location']
df2['attendence'] = df2['observed_attendance']=='yes'

df3 = pd.get_dummies(df2['candidate_job_location'])

df3['travel'] = df2['travel']
df3['hometown'] = df2['hometown']

df3.to_csv('data/millie_cols.csv')
