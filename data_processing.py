import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#importing the dataset
dataset = pd.read_csv('LBW_Dataset.csv')

#replacing nan in Age,Delivery phase,Education,Residence,Community,BP,IFA with mode(most frequent value) 
dataset['Age'].fillna(dataset['Age'].mode()[0],inplace=True)
dataset['Delivery phase'].fillna(dataset['Delivery phase'].mode()[0],inplace=True)
dataset['Education'].fillna(dataset['Education'].mode()[0],inplace=True)
dataset['Residence'].fillna(dataset['Residence'].mode()[0],inplace=True)
dataset['Community'].fillna(dataset['Community'].mode()[0],inplace=True)
dataset['BP'].fillna(dataset['BP'].mode()[0],inplace=True)
dataset['IFA'].fillna(dataset['IFA'].mode()[0],inplace=True)

#replacing nan in weight with minimum value
dataset['Weight'].fillna(dataset['Weight'].min(),inplace=True)

#replacing nan in HB with mean
dataset['HB'].fillna(dataset['HB'].mean(),inplace=True)

#function to normalize a column
def normalize(data):
    n = len(data)
    st= data.std()
    mean = sum(data) / n
    deviations = [(x - mean)/st for x in data]
    return deviations

#normalizing age and weight
dataset['Age']=normalize(dataset['Age'])
dataset['Weight']=normalize(dataset['Weight'])

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Scaling the dataset
sc = StandardScaler()
X = sc.fit_transform(X)

#converting to dataframe to be able to add a column and write to a csv file(the final-pre-processed data)
data=pd.DataFrame(data=X,columns=dataset.columns[0:9])
data['Result']=y
data.to_csv('pre_processed.csv',index=False)

