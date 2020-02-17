
#Loading the libraries and the data
import numpy as np
import pandas as pd
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


wine = pd.read_csv("G:/Assignments/Naive Bayes/Data sets for pratice/winequality.csv")
wine.head()

# Data pre-processing
wine['type'].value_counts().T #

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
encoded_type = encoder.fit_transform(wine.type.values.reshape(-1,1))

wine['encoded_type'] = encoded_type
wine['encoded_type'] = wine['encoded_type'].astype('int64')

wine = wine.drop('type', axis=1)
wine.head()

#Now we check for missing values
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
          # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

missing_values_table(wine)

wine = wine.dropna()

wine['quality'].value_counts().T

#7 of 9 categories are represented. To simplify these, 
#they are now grouped into just 3 categories (1-4, 5-7 and 8-9).

def new_quality_ranking(df):

    if (df['quality'] <= 4):
        return 1
    
    elif (df['quality'] > 4) and (df['quality'] < 8):
        return 2
              
    elif (df['quality'] <= 8):
        return 3

wine['new_quality_ranking'] = wine.apply(new_quality_ranking, axis = 1)
wine = wine.drop('quality', axis=1)
wine = wine.dropna()
wine.head()

wine['new_quality_ranking'].value_counts().T

# Naive Bayes in scikit-learn

# Binary Classification
x = wine.drop('encoded_type', axis=1)
y = wine['encoded_type']

from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(trainX, trainY)

y_pred = gnb.predict(testX)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred))) #Accuracy: 0.97

confusion_matrix(testY,y_pred) # GaussianNB model
pd.crosstab(testY.values.flatten(),y_pred) # confusion matrix using 
np.mean(y_pred==testY.values.flatten()) # 0.96%

# Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB(binarize=0.0)

bnb.fit(trainX, trainY)

y_pred = bnb.predict(testX)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred))) #Accuracy: 0.76

#Multiple Classification
x = wine.drop('new_quality_ranking', axis=1)
y = wine['new_quality_ranking']
 
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

#Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

mnb.fit(trainX, trainY)

y_pred = mnb.predict(testX)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred))) #Accuracy: 0.93

