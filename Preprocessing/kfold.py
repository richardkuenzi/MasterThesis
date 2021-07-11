import pandas as pd
from sklearn.model_selection import StratifiedKFold

name = '00.0010'

df = pd.read_csv('C:/Users/richa/Dropbox/FHNW/MSc/Thesis/Dataset/' + name + '/' + name + '.csv')
df.head()
df = df.sample(frac=1).reset_index(drop=True)

skf = StratifiedKFold(n_splits=5)
target = df.loc[:,name]

fold_no = 1
for train_index, val_index in skf.split(df, target):
    train = df.loc[train_index,:]
    val = df.loc[val_index,:]
    train.to_csv('C:/Users/richa/Dropbox/FHNW/MSc/Thesis/Dataset/' + name + '/' + 'train_fold_' + str(fold_no) + '.csv')
    val.to_csv('C:/Users/richa/Dropbox/FHNW/MSc/Thesis/Dataset/' + name + '/' + 'val_fold_' + str(fold_no) + '.csv')
    fold_no += 1

i = 1
for i in range(1, 6):
    filename = "val_fold_" +str(i)+ ".csv"
    df2 = pd.read_csv(filename)
    first_column = df2.columns[0]
    df2 = df2.drop([first_column], axis=1)
    df2.to_csv(filename, index=False)
    i += 1

j = 1
for j in range(1, 6):
    filename = "train_fold_" +str(j)+ ".csv"
    df3 = pd.read_csv(filename)
    first_column = df3.columns[0]
    df3 = df3.drop([first_column], axis=1)
    df3.to_csv(filename, index=False)
    j += 1