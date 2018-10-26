import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    pd.set_option('display.max_columns', 30)
    df = pd.read_csv('data/interview.csv')

    #EDA and clean columns interview type, gender, industry, obtained permission
    df.rename(columns={'Have you obtained the necessary permission to start at the required time': 'obtained_permission',
                            'Interview Type': 'interview_type', 'Gender': 'gender', 'Industry': 'industry'}, inplace=True)

    df2 = df[['industry', 'interview_type', 'gender', 'obtained_permission']].copy()
    #df.dropna() #drop nan columns
    #df.dtypes #check the type of each column
    #df2['interview_type'].unique()
    #df2['industry'].value_counts())

    #change other categories to yes and no in obtained_permission
    df2['obtained_permission'].replace(to_replace=['Yes','yes'],value='yes', inplace=True)
    df2['obtained_permission'].replace(to_replace=['No','NO'],value='no', inplace=True)
    df2['obtained_permission'].replace(to_replace=['Not yet','Yet to confirm'],value='no', inplace=True)
    df2['obtained_permission'].replace(to_replace=['nan','Na'],value=None, inplace=True)

    #change industry categories
    df2['industry'].replace(to_replace=['IT Services','IT Products and Services', 'IT'], value='IT', inplace=True)

    #change interview type categories
    df2['interview_type'].replace(to_replace=['Scheduled Walkin','Scheduled Walk In', 'Sceduled walkin'],value='scheduled', inplace=True)
    df2['interview_type'].replace(to_replace=['Scheduled '],value='scheduled', inplace=True)
    df2['interview_type'].replace(to_replace=['Walkin ', 'Walkin'],value='walk-in', inplace=True)

    #convert columns to dummy variables
    df3 = df2.copy()
    df3.rename(columns={'interview_type': 'interview_type_scheduled', 'gender': 'male', 'obtained_permission': 'obtained_permission_yes'}, inplace=True)
    df3['interview_type_scheduled'] = df3['interview_type_scheduled'].map({'scheduled': 1, 'walk-in': 0})
    df3['male'] = df3['male'].map({'Male': 1, 'Female': 0})
    df3['obtained_permission_yes'] = df3['obtained_permission_yes'].map({'yes': 1, 'no': 0})

    #check info
    print(df3['industry'].value_counts())
    print(df3['interview_type_scheduled'].value_counts())
    print(df3['male'].value_counts())
    print(df3['obtained_permission_yes'].value_counts())

    #convert and export as csv - finialize columns
    ryan_cols = ['interview_type_scheduled','male','obtained_permission_yes']
    df4 = df3[ryan_cols]
    df4.to_csv('data/ryan.csv')
