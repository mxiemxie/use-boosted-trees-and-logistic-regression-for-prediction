# Read Train data and test data
import pandas as pd
resultPath = 'cognical_project\\data_modeling\\result.csv'
set1Path = 'cognical_project\\data_modeling\\set1.csv'
set2Path = 'cognical_project\\data_modeling\\set2.csv'
set3Path = 'cognical_project\\data_modeling\\set3.csv'
result = pd.read_csv(resultPath)
set1 = pd.read_csv(set1Path)
set2 = pd.read_csv(set2Path)
set3 = pd.read_csv(set3Path)
data = pd.concat([set1, set2, set3], axis = 1)
test = result[-1093:]
test.head()
testData = pd.concat([test, data], axis = 1, join = 'inner')
train = result[ :-1093]
trainData = pd.concat([train, data], axis = 1, join = 'inner')
print('{} rows test data, {} rows train data'.format(testData.shape[0], trainData.shape[0]))
# We have 1093 targets, but only 993 rows for it. It means we lose 100 rows data. So 
# the count of missing data is really important to us. I try to find some relationship 
# between the counts of the missing attributes and the class. Unfortunately, it looks 
# there is not so much relationship between them. Later, I find there are 2086 YES from 
# 4370 rows data. So I will predict the target with missing data No.
trainMissingCount = pd.concat([trainData.isnull().sum(axis = 1), trainData], axis = 1, join = 'inner')
trainMissingCount.rename(columns={0:'MissingValueCount'}, inplace=True)
%matplotlib inline
MissingCountClass = trainMissingCount[['MissingValueCount', 'DelinquencyClass']]
MissingCountClass['DelinquencyClass'] = MissingCountClass['DelinquencyClass'].astype('category')
cat_columns = MissingCountClass.select_dtypes(['category']).columns
MissingCountClass[cat_columns] = MissingCountClass[cat_columns].apply(lambda x: x.cat.codes)
MissingCountClass.plot(x = 'MissingValueCount', y = 'DelinquencyClass', kind = 'scatter')
print('There are {} Yes in {} rows of data'.format(sum(MissingCountClass['DelinquencyClass']), trainData.shape[0]))
# Use PLS to do data visualization of the original data
from sklearn.cross_decomposition import PLSRegression
X = trainData.drop('DelinquencyClass', axis = 1)
Y = trainData[['DelinquencyClass']]
cat_columns = X.select_dtypes(['category']).columns
X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)
Y['DelinquencyClass'] = Y['DelinquencyClass'].astype('category')
cat_columns = Y.select_dtypes(['category']).columns
Y[cat_columns] = Y[cat_columns].apply(lambda x: x.cat.codes)
pls2 = PLSRegression(n_components=2)
pls2.fit(X, Y)
import pylab
pls = PLSRegression(n_components=30)
pls.fit(X, Y)
x_transform, y_transform = pls.transform(X, Y)
y = Y['DelinquencyClass'].astype(str).astype(int)
pylab.scatter(x_transform[:,0], x_transform[:,1], c = y)
pylab.suptitle('First impression of data')