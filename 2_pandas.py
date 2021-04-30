# Pandas

# Pandas is a Python library used for working with data sets.
# It has functions for analyzing, cleaning, exploring, and manipulating data.
# The name "Pandas" has a reference to both "Panel Data", and "Python Data Analysis"

import pandas as pd

# Series    : One Dimentional array with axis labels (like a single column of a 2D-array / DataFrame)
# Dataframe : 2D array with axis labels (row and column) (containing multiple Series)

# Creating Series
sr1 = pd.Series([45, 656, 567, 5, 67, 6])
sr1 = pd.Series({'first': 1, 'second': 2, 'third': 3})   # The keys of the dictionary become the labels of the rows
sr1 = pd.Series([45, 656, 567, 5], index=['first', 'second', 3, 'forth']) # index = [...] (labels...)

# index   : labels of rows    (default : 0, 1, 2, ...)
# columns : labels of columns (default : 0, 1, 2, ...)

# Creating Dataframes
data = {"calories": [420, 380, 390], "duration": [50, 40, 45]}
marks = [
    ['Nayeem', 86],
    ['Sami', 63],
    ['Alif', 53],
]
df = pd.DataFrame(data) # The keys of the dictionary become the label of the columns
df = pd.DataFrame(data, index=['first', 'second', 'third'])
df1 = pd.DataFrame(marks, columns=['Name', 'Marks'])
df2 = pd.DataFrame(marks, index=['first', 'second', 'third'], columns=['Name', 'Marks'])


# Read, Write CSV
df1.to_csv('data\\marks_data1.csv')
df1.to_csv('data\\marks_data2.csv', index=False) # index/row label will not be saved
df1.to_csv('data\\marks_data3.csv', columns=['Marks']) # only 'Marks' column will be saved

df3 = pd.read_csv('data\\marks_data1.csv')

# Working with Exel, Json and others is same....

np_array = df.to_numpy() # DataFrame to numpy array



# Simple Data analysis
marks = [
    ['Nayeem', 301, 89],
    ['Sami', 302, 76],
    ['Alif', 305, 71],
    ['Rabbi', 315, 85],
    ['Kazi', 309, 73],
    ['Toyon', 356, 86],
    ['Faruk', 314, 75],
    ['Oasib', 325, 69],
    ['Fahim', 328, 65],
]
df = pd.DataFrame(marks, columns=['name', 'roll', 'mark'])
df2 = pd.DataFrame(marks)

# printing the whole DataFrame
print(df)
print(df.to_string())

print(df.head()) # returns first 5 rows
print(df.head(4)) # returns first 4 rows

print(df.tail()) # returns last 5 rows
print(df.tail(6)) # returns last 6 rows

print(df.info())     # print the basic info like: columns, dtype, count, memory usage etc.
print(df.describe()) # print count, mean, std, min, 25%, 50%, 75%, max

print(df.index)
print(df.columns)
print(df.shape)



# Copy vs View
newdf = df # this will create a view. 'newdf' will point to 'df'
newdf = df.copy() # this will be a copy. Now, 'newdf' and 'df' is fully apart
# most of the functions like: drop, loc, iloc, reset_index returns a copy
# If you want to do something to the orginial dataframe, pass this : inplace=True



# Accesing and modifying DataFrame
# print(df['mark']) # returns that column (a Series)
# print(df['mark'][4]) # returns element on index 4 in 'marks'

# setting element this way sholud be ignored
# df['mark'][4] = 83

# loc
print(df.loc[4, 'mark'])   # 4 : index  | 'mark' : column
df.loc[4, 'mark'] = 45 # if any given value of index or column is not found, it will create a new one

# selecting a range
print(df.loc[[0, 1, 3, 5], ['roll', 'mark']]) # return elements of 0, 1, 3, 5 index from 'roll', 'mark' columns
print(df.loc[:, ['mark']])                    # return elements of all index from 'mark' column
print(df.loc[[0, 5], :])                      # return elements of 0 to 5 index from all columns
print(df.loc[2:5, 'name':'mark'])             # return elements of 0 to 5 index from 'name' to 'mark' columns

# iloc
# same as loc, just here we need to pass exact index/column, not the label
print(df.iloc[4, 2])
df.iloc[4, 2] = 45
print(df.iloc[:, [1, 2]])

# df.<loc|iloc>[<index|list_of_index|range_of_index>, <column|list_of_column|range_of_column>]    | he he he he he...

# conditional loc/iloc (query)
df.loc[df['mark'] > 75]
df.loc[(df['mark'] > 75) & (df['roll'] < 320)]




# Drop (delete)
newdf = df.drop(2) # default axis = 0 | # delete index 2 (3 no. row) 
newdf = df.drop('roll', axis=1) # delete 'roll' column
newdf = df.drop(['roll', 'name'], axis=1) # delete 'roll' and 'name' column

# After dropping or anything, the index/column can get messy, to reset that
newdf2 = newdf.reset_index() # this will not drop the previous index, it will name that a new column
newdf2 = newdf.reset_index(drop=True)


# Get Value Counts
mark_counts = df['mark'].value_counts()


# Column rename:
newdf = df.rename(columns = {'roll': 'roll_number', 'mark': 'result'}, inplace=False)


# Some functions
x = df["mark"].max()
x = df["mark"].min()
x = df["mark"].mean()
x = df["mark"].median()
x = df["mark"].std()
x = df["mark"].mode()[0]

x = df["mark"].value_counts() # count all value without null
x = df["mark"].value_counts(dropna=False) # count all value


# Cleaning data
data = [
    ['Nayeem', 301, 86, 73,   96],
    ['Masuda', 302, 75, 71,   76],
    [None,     305, 84, None, 65],
    ['Kawser', 308, 83, 79,   None],
    ['Lamima', 303, 75, 81,   77],
]
df = pd.DataFrame(data, columns=['name', 'roll', 'bangla', 'english', 'math'])

print(df.isnull()) # check null

new_df = df.dropna() # Return a new Data Frame with no empty cells
df.fillna(100, inplace = True) # Replace NULL values with the number 100
df["math"].fillna(130, inplace = True) # Replace NULL values with the number 100 in 'math' column
df["math"].fillna(df['math'].mean(), inplace = True) # Replace NULL values with the mean of 'math' column


# wrong format
dates = [
    ['2020/12/20', 97,  125],
    ['2020/12/21', 108, 131],
    [None,         100, 119],
    ['2020/12/23', 130, 101],
    ['2020/12/25', 102, 126],
]
df = pd.DataFrame(dates)

df[0] = pd.to_datetime(df[0])
df.dropna(subset=[0], inplace=True) # to dropna from a column, use subset
df.reset_index(drop=True, inplace=True)
print(df)


# Duplicate Value
print(df.duplicated()) # Returns True for every row that is a duplicate
df.drop_duplicates(inplace=True)


# Pandas Correlations (Finding Relationships)
data = [
    ['Nayeem', 301, 86, 73,   96],
    ['Masuda', 302, 75, 71,   76],
    ['Someone',305, 84, 73,   96],
    ['Kawser', 308, 83, 79,   94],
    ['Lamima', 303, 75, 81,   77],
]
df = pd.DataFrame(data, columns=['name', 'roll', 'bangla', 'english', 'math'])
print(df.corr())

# this represent increment/decrement ratio with two rows taht is -1 to 1
# 1 or close to it means a good relationship
# -1 or it's close means a good relationship but the opposite (if one increase, other one decrease)
# 0.2, -0.35 ... these are not good relationship

#              roll    bangla   english      math
# roll     1.000000  0.234505  0.465440  0.372427
# bangla   0.234505  1.000000 -0.211876  0.987103
# english  0.465440 -0.211876  1.000000 -0.142562
# math     0.372427  0.987103 -0.142562  1.000000

# We ca see a good relationship in bangla and math
# the centre 1.000 is for roll increse same with roll, because roll is same to roll
# other values determeans relationships, like: roll * english --> 0.465440
# this means, if roll is increased, 46.5440% chance to increase english number
