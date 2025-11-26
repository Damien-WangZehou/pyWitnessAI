import os.path
import pandas as pd
import ast


def flatten_keys(structure):
    keys = flatten_recursive(structure)
    return [k[0:-4] for k in keys]


def flatten_data(structure):

    data = []

    if type(structure) == list:
        for i, e in zip(range(0, len(structure)), structure):
            for d in flatten_data(e):
                data.append(d)
    elif type(structure) == dict:
        for k in structure:
            for d in flatten_data(structure[k]):
                data.append(d)
    else:
        data = [structure]

    return data


def flatten_recursive(structure):

    keys = []

    if type(structure) == list:
        for i, e in zip(range(0, len(structure)), structure):
            for dk in flatten_recursive(e):
                keys.append(str(i)+"_"+dk)
    elif type(structure) == dict:
        for k in structure:
            for dk in flatten_recursive(structure[k]):
                keys.append(k+"_"+dk)
    else:
        keys = ['val']

    return keys


#  A class to flatten specific column, which will make the data file more beautiful
class DataFlattener:
    def __init__(self, dataframe, column_to_be_processed=None, preprocess_lists=True):
        if column_to_be_processed is None:
            column_to_be_processed = ['mtcnn_confidence', 'mtcnn_coordinates', 'similarity_facenet_distance']

        self.column = column_to_be_processed
        self.df = dataframe
        if preprocess_lists:
            self.preprocess_list_columns(self.column)

    def preprocess_list_columns(self, column_to_be_processed):
        #  Add all columns here that need conversion from string to list
        list_columns = column_to_be_processed
        for col in list_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self.safe_literal_eval)

    @staticmethod
    def safe_literal_eval(row):
        try:
            return ast.literal_eval(row)
        except (ValueError, SyntaxError):
            return []

    def flatten_column(self, column_name, new_column_template):
        #  Extract the maximum number of items in the nested lists of the specified column
        max_items = self.df[column_name].apply(lambda x: len(x)).max()

        #  Create new columns for each item
        for i in range(max_items):
            #  New column names
            new_col_name = f'{new_column_template}_{i+1}'
            #  Assign the values from the nested list
            self.df[new_col_name] = self.df[column_name].apply(lambda x: x[i] if i < len(x) else None)

        #  Drop the original column
        self.df.drop(column_name, axis=1, inplace=True)

    def flatten_multidimensional_column(self, column_name, new_column_template):
        #  Get the max dimension
        dimensions = self.df[column_name].apply(lambda x: (len(x), len(x[0]) if x else 0)).max()
        print(f'dimension of this column is {dimensions}')

        #  Create new columns for each item in the 2D list
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                new_col_name = f'{new_column_template}_{i+1}_{j+1}'
                self.df[new_col_name] = self.df[column_name].apply(
                    lambda x: x[i][j] if i < len(x) and j < len(x[i]) else None)

        self.df.drop(column_name, axis=1, inplace=True)

    def flatten(self):
        #  We should know which column needs to be flattened
        self.flatten_column('mtcnn_confidence', 'mtcnn_confidence')
        self.flatten_multidimensional_column('mtcnn_coordinates', 'mtcnn_coordinates')
        self.flatten_multidimensional_column('similarity_facenet_distance',
                                             'similarity_facenet_distance')

        return self.df


# # How to use the Flattener
# df = pd.read_csv('D:/MscPsy/Data/DataForTest/results/analyzed_data.csv')
# flattener = DataFlattener(df)
# flat_df = flattener.flatten()
#
# # Save the flattened DataFrame back to a CSV
# flat_df.to_csv(os.path.join('D:/MscPsy/Data/DataForTest/results', 'flattened_data_changedFaces.csv'), index=False)