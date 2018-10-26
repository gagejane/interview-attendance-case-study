import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.axes as ax

df_orig = pd.read_csv('data/interview.csv')
pd.options.display.max_columns = 28
# df.head()

df = df_orig.copy()
cols = df_orig.columns.tolist()
cols = [col.replace(' ', '_') for col in cols]
cols = [col.replace('._', '_') for col in cols]
df.columns = cols
#
# print(df['Hope_there_will_be_no_unscheduled_meetings'].value_counts())
# print(df['Can_I_Call_you_three_hours_before_the_interview_and_follow_up_on_your_attendance_for_the_interview'].value_counts())
# print(df['Can_I_have_an_alternative_number/_desk_number_I_assure_you_that_I_will_not_trouble_you_too_much'].value_counts())
# print(df['Have_you_taken_a_printout_of_your_updated_resume_Have_you_read_the_JD_and_understood_the_same'].value_counts())
# print(df['Observed_Attendance'].value_counts())

df.rename(columns={'Hope_there_will_be_no_unscheduled_meetings': 'unscheduled_meetings',
 'Can_I_Call_you_three_hours_before_the_interview_and_follow_up_on_your_attendance_for_the_interview': 'call_before', 'Can_I_have_an_alternative_number/_desk_number_I_assure_you_that_I_will_not_trouble_you_too_much': 'alternate_phone', 'Have_you_taken_a_printout_of_your_updated_resume_Have_you_read_the_JD_and_understood_the_same': 'print_resume', 'Observed_Attendance': 'observed_attendance', 'Industry':'industry' }, inplace=True)


df['unscheduled_meetings'].replace(to_replace=['Yes','yes'],value=1, inplace=True)
df['unscheduled_meetings'].replace(to_replace=['No'],value=0, inplace=True)
df['unscheduled_meetings'].replace(to_replace=['Na', 'Not sure', 'Not Sure', 'cant Say'],value=np.NaN, inplace=True)

df['call_before'].replace(to_replace=['Yes','yes'],value=1, inplace=True)
df['call_before'].replace(to_replace=['No', 'No Dont'],value=0, inplace=True)
df['call_before'].replace(to_replace=['Na'],value=np.NaN, inplace=True)

df['print_resume'].replace(to_replace=['Yes','yes'],value=1, inplace=True)
df['print_resume'].replace(to_replace=['No', 'No- will take it soon', 'Not Yet', 'Not yet'],value=0, inplace=True)
df['print_resume'].replace(to_replace=['Na', 'na'],value=np.NaN, inplace=True)

df['alternate_phone'].replace(to_replace=['Yes','yes'],value=1, inplace=True)
df['alternate_phone'].replace(to_replace=['No', 'No I have only thi number'],value=0, inplace=True)
df['alternate_phone'].replace(to_replace=['Na', 'na'],value=np.NaN, inplace=True)

df['observed_attendance'].replace(to_replace=['Yes','yes','yes', 'yes '],value=1, inplace=True)
df['observed_attendance'].replace(to_replace=['No', 'NO', 'no', 'no ', 'No '],value=0, inplace=True)
# df['observed_attendance'].replace(to_replace=['Na', 'na'],value=np.NaN, inplace=True)

df['industry'].replace(to_replace=['IT Services','IT Products and Services', 'IT'], value='IT', inplace=True)
# df2.isnull().sum()

print(df['observed_attendance'].value_counts())

# print(df['unscheduled_meetings'].value_counts()/len(df)*100)
# print(df['call_before'].value_counts()/len(df)*100)
# print(df['alternate_phone'].value_counts()/len(df)*100)
# print(df['print_resume'].value_counts()/len(df)*100)
print(df['observed_attendance'].value_counts()/len(df)*100)
#
# print(df['unscheduled_meetings'].value_counts(dropna=True)/len(df)*100)
# print(df['call_before'].value_counts(dropna=True)/len(df)*100)
# print(df['alternate_phone'].value_counts(dropna=True)/len(df)*100)
# print(df['print_resume'].value_counts(dropna=True)/len(df)*100)
# print(df['observed_attendance'].value_counts(dropna=True)/len(df)*100)

data_dict = {'Hope_there_will_be_no_unscheduled_meetings': 'unscheduled_meetings',
 'Can_I_Call_you_three_hours_before_the_interview_and_follow_up_on_your_attendance_for_the_interview': 'call_before', 'Can_I_have_an_alternative_number/_desk_number_I_assure_you_that_I_will_not_trouble_you_too_much': 'alternate_phone', 'Have_you_taken_a_printout_of_your_updated_resume_Have_you_read_the_JD_and_understood_the_same': 'print_resume', 'Observed_Attendance': 'observed_attendance', 'Are you clear with the venue details and the landmark.': 'venue_knowledge','Has the call letter been shared': 'seen_call_letter', 'Expected Attendance': 'expected_attendance','Marital Status': 'marital_status','Position to be closed': 'position', 'Have you obtained the necessary permission to start at the required time': 'obtained_permission', 'Interview Type': 'interview_type','Gender':'gender', 'Industry':'industry'}

data_dict_final = {v: k for k, v in data_dict.items()}



df_biz_type = df['industry']
df_attend = df['observed_attendance']

print(pd.crosstab(df_biz_type,df_attend, normalize='index'))


# jane_cols = ['unscheduled_meetings','call_before','alternate_phone','print_resume','observed_attendance']
# df = df[jane_cols]
# df.to_csv('data/jane_cols.csv')


zipped = [('May I call before?', 0.0010200779708488584), ('Unscheduled meetings', 0.006972123524095326), ('Cochin', 0.008191334052088139), ('Hosur', 0.010302232462681216), ('Noida', 0.012817272901900862), ('Visakapatinam', 0.013844654065914352), ('Gurgaon', 0.016211527503406507), ('Print out resume', 0.019034210813727328), ('Have venue details', 0.022477199429498503), ('Gave alternate phone', 0.03024609507022577), ('Bangalore', 0.035858830774309104)
, ('Chennai', 0.03610515923080236), ('Travel for interview?', 0.03844664172690597), ('Married', 0.06338354261654086), ('Call letter seen', 0.06730774430453229), ('Interview scheduled', 0.06798446330760234), ('Man', 0.07798043013943225), ('hometown', 0.0809007272814148), ('Permission', 0.0982884590519811), ('Type of skillset', 0.11177183522742581), ('Expected to attend', 0.18085543854466618)]


unzipped = zip(*zipped)
variables, score = unzipped

# plt.bar(variables, score)
# labels = variables
# plt.xticks(rotation=70, horizontalalignment='right')
# plt.title('Influence Scores', weight='bold', size=14)
# plt.xlabel('Variable labels', weight='bold')
# plt.ylabel('Scores', weight='bold')
# plt.tight_layout()


zipped2 = [('Married', 0.06338354261654086), ('Call letter seen', 0.06730774430453229), ('Interview scheduled', 0.06798446330760234), ('Man', 0.07798043013943225), ('hometown', 0.0809007272814148), ('Permission', 0.0982884590519811), ('Type of skillset', 0.11177183522742581), ('Expected to attend', 0.18085543854466618)]

unzipped2 = zip(*zipped2)
variables2, score2 = unzipped2

plt.bar(variables2, score2)
labels = variables2
plt.xticks(rotation=70, horizontalalignment='right')
plt.title('Influence Scores', weight='bold', size=14)
plt.xlabel('Variable labels', weight='bold')
plt.ylabel('Scores', weight='bold')
plt.tight_layout()
# plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
# plt.xticks(np.arange(0, 1, step=0.2))
# ax.xaxis.set_tick_params(width=5)
# plt.show()
# plt.savefig('Infl_scores2')
# def codebook(data_dict):
#     for key, value in data_dict.items():

#
# Goal: Create a model that predicts observed attendance rate.
#
# Target: Observed Attendance
#
# Features:
# Industry
# Location
# Position to be closed
# Nature of Skillset
# Interview Type
# Gender
# Candidate current location
# Obtained permission
# Unshedule meeting
# Call you
# Call alternate
# Taken printout
# Clear with vanue
# Call letter hsared
# Expected Attendance
# Marical status
#
# Drop:
# Date? ugghhh
# Client name
# Unamed 23, 24, 25, 26, 27

# Tasks: clean variable names?
# e.g., Can I Call you three hours before the interview and follow up on your attendance for the interview --> CallBefore
