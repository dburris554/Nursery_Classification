import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree

column_names = ('parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'label')

def get_file_as_dataframe(message, has_labels):
    file = input(message)
    df = pd.read_csv(file)
    # minor data adjustments
    df.columns = list(column_names) if has_labels else list(column_names[:-1])
    to_english = {'1': 'one', '2': 'two', '3': 'three', 'more': 'more'}
    df['children'].replace(to_english, inplace=True)
    return df
    
def ordinal_to_numeric(dataframe, has_labels):
    encoder = OrdinalEncoder()
    dataframe = pd.DataFrame(encoder.fit_transform(dataframe))
    size = 9 if has_labels else 8
    dataframe.columns = [str(i) for i in range(size)]
    return dataframe, encoder

def numeric_to_ordinal(dataframe, encoder):
    df = pd.DataFrame(encoder.inverse_transform(dataframe))
    df.columns = list(column_names)
    from_english = {'one': '1', 'two': '2', 'three': '3', 'more': 'more'}
    df['children'].replace(from_english, inplace=True)
    return df

# get train data
train_df = get_file_as_dataframe('Input relative path to training data file (must be in same folder):\n', True)
train_df, feat_label_enc = ordinal_to_numeric(train_df, True)

# separate features from labels
train_feats = train_df[[str(i) for i in range(8)]]
train_labels = train_df[['8']]

# train decision tree classifier
my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(train_feats, train_labels)

# get test data
test_df = get_file_as_dataframe('Input relative path to test data file (must be in same folder):\n', False)
test_df, _ = ordinal_to_numeric(test_df, False)

# use tree classifier to predict and store test labels
test_df['8'] = my_tree.predict(test_df)

# convert back to ordinal data
test_df = numeric_to_ordinal(test_df, feat_label_enc)

# print test data and labels
for row in test_df.itertuples(name='Nursery'):
    print(row)