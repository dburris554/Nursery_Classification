import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree

train_file = input("Input relative path to training data file (must be in same folder):\n")
train_df = pd.read_csv(train_file)
train_df.columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
to_english = {'1': 'one', '2': 'two', '3': 'three', 'more': 'more'}
train_df['children'].replace(to_english, inplace=True)
train_ord = OrdinalEncoder()
train_num_df = pd.DataFrame(train_ord.fit_transform(train_df))
train_num_df.columns = [str(i) for i in range(9)]
train_data = train_num_df[[str(i) for i in range(8)]]
train_labels = train_num_df[['8']]

my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(train_data, train_labels)

test_file = input("Input relative path to test data file (must be in same folder):\n")
test_df = pd.read_csv(test_file)
test_df.columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']
test_df['children'].replace(to_english, inplace=True)
test_ord = OrdinalEncoder()
test_num_df = pd.DataFrame(test_ord.fit_transform(test_df))
test_num_df.columns = [str(i) for i in range(8)]
test_num_df['8'] = my_tree.predict(test_num_df)
test_num_df.columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
res_df = pd.DataFrame(train_ord.inverse_transform(test_num_df))
res_df.columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'guess']
for row in res_df.itertuples(name='Nursery'):
    print(row)