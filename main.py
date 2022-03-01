import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree

def get_file_as_dataframe(message, has_labels):
    file = input(message)
    df = pd.read_csv(file)
    # minor data adjustments
    cols = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']
    if has_labels:
        cols.append('label')
    df.columns = cols
    to_english = {'1': 'one', '2': 'two', '3': 'three', 'more': 'more'}
    df['children'].replace(to_english, inplace=True)
    return df
    
# get training data and store in DataFrame
train_file = input("Input relative path to training data file (must be in same folder):\n")
train_df = pd.read_csv(train_file)

# minor data adjustments
train_df.columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'label']
to_english = {'1': 'one', '2': 'two', '3': 'three', 'more': 'more'}
train_df['children'].replace(to_english, inplace=True)

# convert from ordinal data to numeric data
feat_label_ord = OrdinalEncoder()
train_num_df = pd.DataFrame(feat_label_ord.fit_transform(train_df))
train_num_df.columns = [str(i) for i in range(9)]

# separate features from labels
train_feat = train_num_df[[str(i) for i in range(8)]]
train_labels = train_num_df[['8']]

# train decision tree classifier
my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(train_feat, train_labels)

# get test data and store in DataFrame
test_file = input("Input relative path to test data file (must be in same folder):\n")
test_df = pd.read_csv(test_file)

# minor data adjustments
test_df.columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']
test_df['children'].replace(to_english, inplace=True)

# convert from ordinal data to numeric data
feat_ord = OrdinalEncoder()
test_num_df = pd.DataFrame(feat_ord.fit_transform(test_df))
test_num_df.columns = [str(i) for i in range(8)]

# use tree classifier to predict and store test labels
test_num_df['8'] = my_tree.predict(test_num_df)

# convert from numeric data to ordinal data
res_df = pd.DataFrame(feat_label_ord.inverse_transform(test_num_df))
res_df.columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'label']

# print test data and labels
for row in res_df.itertuples(name='Nursery'):
    print(row)