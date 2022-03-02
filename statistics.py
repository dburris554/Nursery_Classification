import pandas as pd

column_names = ('parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'label')

def get_file_as_dataframe(message):
    file = input(message)
    df = pd.read_csv(file)
    df.columns = list(column_names)
    return df

# get test data with labels
given_df = get_file_as_dataframe('Provide labeled test data file name:\n')

# get project-generated data
my_df = get_file_as_dataframe('Provide project-generated file name:\n')

my_df['match'] = given_df['label'] == my_df['label']
cnt_true = my_df['match'].value_counts()[True]
cnt_total = my_df['match'].value_counts().sum()
print(f'Accuracy: {cnt_true} / {cnt_total} = {cnt_true / cnt_total * 100} %')
