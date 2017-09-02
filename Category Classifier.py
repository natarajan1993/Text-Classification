"""This was a machine learning project for text analytics and classification. The objective was to find the most important keywords associated with whether a protection agreement was sold to the customer or not during the sale of appliances. The data is chats between sales representatives and customers online. I achieved the goals using scikit-learn ML algorithms. I can't upload the data I used because of an NDA but this is the code that I used. I pre-processed the text using Pandas and regular expressions. Then I used 4 different algorithms to compare performance and in the end decided on logistic regression. The algorithms were Logistic Regression, Support Vector Machines, Support Vector Machines with Stochastic Gradient Descent and Perceptron. The structure of the data used was:

Protection Agreement bought? (0/1)		Order ID		Chat
0/1										ID 1			LINE 1
0/1										ID 1			LINE 2
0/1										ID 1			LINE 3
.
.
.
"""
import re
import nltk
import time
import random
import codecs
import itertools
import numpy as np
import pandas as pd

from nltk import word_tokenize          
from nltk.stem.snowball import SnowballStemmer

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from IPython.display import Image
pd.options.display.max_colwidth = 500

# Filenames can be Agent_Closed_ or Insession_Closed_
file_name = 'Agent_Closed_'
ngram = 'bigram'
num_features = 20
ngram_range = tuple()
if ngram is 'bigram':
    ngram_range = (2,2) 
elif ngram is 'trigram':
    ngram_range = (3,3) 
elif ngram is 'quadgram':
    ngram_range = (4,4) 
else:
    ngram_range = (1,1) 
ngram_range

"""The Lemmatizer copied from sklearn's documentation
http://scikit-learn.org/stable/modules/feature_extraction.html
I used Snowball Stemmer instead of WordNetLemmatizer"""
class LemmaTokenizer(object):
    def __init__(self):
        self.sbs = SnowballStemmer("english")
    def __call__(self, doc):
        return [self.sbs.stem(t) for t in word_tokenize(doc)]

"""Custom function that zips the coefficients with the feature names and prints the top 20"""		
def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

"""Pre-processing"""
df = pd.read_csv(r'C:\Users\Natarajan\Documents\Jupyter notebooks\Sears\Datasets\{}Order_Chat.csv'.format(file_name),encoding = 'ascii',sep = ',')
df.dropna(axis = 0, inplace = True)
df['Chat'].drop_duplicates() 
df.drop(['InteractionID','AgentID','AgentName','Trs_Dt','Closing_type','Member_Ind','Sequence','Order_Amt','Role'],axis = 1, inplace = True) # Drop unnecessary columns
df = df.groupby(['OrdId','PA_Ind'])['Chat'].apply(r' '.join).reset_index() # Convert the dataframe from line by line of each chat to entire chats per row
df['Chat'] = df['Chat'].str.replace(r'(\r\n|\n|\r)',' ') # Replace all \r \r\n and \n with a space
df['Chat'] = df['Chat'].str.lower() # Convert the entire series to lowercase

"""More pre-processing"""
df['Chat'] = df['Chat'].str.replace(r'[^a-zA-Z]',
                       r' ') # Replace anything but letters and 3 and 5 with a space. Numbers showed up in the output without this
                             # Did this to preserve the 3 year and 5 year keywords but remove if needed
                             # Accuracy lowered if I did this
df['Chat'] = df['Chat'].str.replace(r'proection',
                       r'protection') # This misspelling showed up on the output so I explicitly replaced it

"""All the variations
    - Year or years or yrs or Protection or protections
        - Followed by one or more spaces
            - protection
            - protections
            - plan
            - plans
            - warranty
            - warranties
            - agreement
            - agreements"""

df['Chat'] = df['Chat'].str.replace(r'(\b(years?|protections?|yrs?)[ ]{1,}(protections?|plans?|warrant(y|ies)|agreements?)?\b)',
                       r' YearX_PlanX ')

"""________Start of Machine learning part________"""

"""Do the train test split where the features are the chats and the output is whether the insurance was bought or not"""
X = df.Chat 
y = df.PA_Ind
X_train, X_test, y_train, y_test = train_test_split(X, y) # Do the train test split with default parameters

"""Remove any stopwords as needed"""
stopwords_file = r'C:\Users\Natarajan\Documents\Jupyter notebooks\Sears\Misc\stopwords.txt'
custom_stopwords = set(codecs.open(stopwords_file, 'r', 'latin-1').read().splitlines())
default_stopwords = set(nltk.corpus.stopwords.words('english'))
#Combine custom and default stopwords with bitwise OR
all_stopwords = default_stopwords | custom_stopwords
my_additional_stop_words = ('<','>','span','/span','dir=','ltr','href=','td','br','sure','tr','dojoxgridcontent','class',
                            'padding','0px') # Add any additional stopwords we dont want and update the list
stop_words = all_stopwords.union(my_additional_stop_words)

'''Initialize the countvectorizer and set the number of n-grams needed. Much more easier step'''
vect = CountVectorizer(stop_words = stop_words,ngram_range=ngram_range,encoding = 'latin-1',binary=True,tokenizer = LemmaTokenizer(),strip_accents = 'ascii') # Set the tokenizer with our custom class defined at the top

X_train_dtm = vect.fit_transform(X_train) # Convert to sparse matrix of document terms
X_test_dtm = vect.transform(X_test)

"""Create the different algorithms"""
logreg = LogisticRegression(solver = 'sag',max_iter = 10000,intercept_scaling=1000)
svm = LinearSVC()

"""After playing with the loss functions, the default hinge gave the best accuracy. See the last cell for tests.
Changing the learning rate alpha didn't change much but only exponentially increased the coefficients given to each feature.
I set number of iterations to 100 just to be sure of convergence. Default is 5.
Always set regularization to l2 as it is usually superior to l1 
http://www.chioka.in/differences-between-the-l1-norm-and-the-l2-norm-least-absolute-deviations-and-least-squares/
"""
sgd = SGDClassifier() 
ptrn = Perceptron() # http://scikit-learn.org/stable/modules/linear_model.html#perceptron

"""Fit the different models with the training data"""
ptrn.fit(X_train_dtm,y_train)
sgd.fit(X_train_dtm,y_train)
logreg.fit(X_train_dtm,y_train)
svm.fit(X_train_dtm, y_train)

"""Get the predicted output for each model"""
y_pred_class_logreg = logreg.predict(X_test_dtm)
y_pred_class_svm = svm.predict(X_test_dtm)
y_pred_class_sgd = sgd.predict(X_test_dtm)
y_pred_class_ptrn = ptrn.predict(X_test_dtm)

"""Print accuracies for each model"""
print("Logistic Regression Accuracy:",np.round(metrics.accuracy_score(y_test, y_pred_class_logreg) * 100,2))
print("Support Vector Machine Accuracy:",np.round(metrics.accuracy_score(y_test, y_pred_class_svm) * 100,2))
print("SVM with Stochastic Gradient Descent Accuracy:",np.round(metrics.accuracy_score(y_test, y_pred_class_sgd) * 100,2))
print("Perceptron Accuracy:",np.round(metrics.accuracy_score(y_test, y_pred_class_ptrn) * 100,2))


print("Logistic Regression Accuracy:",np.round(metrics.accuracy_score(y_test, y_pred_class_logreg) * 100,2))
print("Logistic Regression Features")
show_most_informative_features(vect,logreg,num_features)

print("Support Vector Machine Accuracy:",np.round(metrics.accuracy_score(y_test, y_pred_class_svm) * 100,2))
print("Support Vector Machine Features")
show_most_informative_features(vect,svm,num_features)

print("SVM with Stochastic Gradient Descent Accuracy:",np.round(metrics.accuracy_score(y_test, y_pred_class_sgd) * 100,2))
print("SVM with Stochastic Gradient Descent Features")
show_most_informative_features(vect,sgd,num_features)

print("Perceptron Accuracy:",np.round(metrics.accuracy_score(y_test, y_pred_class_ptrn) * 100,2))
print("Perceptron Features")
show_most_informative_features(vect,ptrn,num_features)

"""________End of Machine learning part________"""

"""Purpose of this section is for each chat ID find the counts of the top 20 keywords and multiply them together to get the final 'score' of each chat"""

"""Create a stemmer object for the english corpus and then create a new column 'stemmed' where the raw chat is converted to 
   a stemmed version of the same text. The 3 lines do that"""
stemmer = SnowballStemmer('english')
df['stemmed'] = df["Chat"].apply(lambda x: [stemmer.stem(y) for y in word_tokenize(x)])
df['stemmed'] = df['stemmed'].apply(lambda x: ' '.join(x))
# df.drop(['Contains'],inplace = True, axis = 1)

"""Same function as above but returns the top keywords instead of printing them"""
def return_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    return top

def find_counts(tdf):
    """This is a complex line. The logic is:
            For each row in the dataframe that is received as input for the function
                Get the row which matches the order id between the input dataframe and our main dataframe
                    In the 'stemmed' column
                        use the str.count method to count the number of occurrences of our regex pattern
                            split the value in the 'keyword' column by space and join again with the regex pattern .*?
                                - Means a lazy match in regex
                                - Without this the number of matches is very low
                                - Means get the smallest match with the start at the first word and end at last word with 
                                  any number of characters in between
                            return values[0] which is a pandas series method that returns the value
                                - otherwise it returns a numpy array object
       Also we are searching in the stemmed chat and not the original"""
    return df[df['OrdId'] == tdf['OrdId']]['stemmed'].str.count('.*?'.join(tdf['Keyword'].split(' '))).values[0]
#     return df[df['OrdId'] == tdf['OrdId']]['stemmed'].str.count(tdf['Keyword']).values[0]

"""Create 4 lists of the coefficients and feature names to help with calculations"""
coef_1,coef_2,fn_1,fn_2 = [],[],[],[]
for (c1, f1), (c2, f2) in return_most_informative_features(vect, logreg, n=num_features):
    coef_1.append(c1) # Coef_1 and fn_1 are the negative coefficients and feature names
    coef_2.append(c2) # Coef_2 and fn_2 are the positive coefficients and feature names
    fn_1.append(f1)
    fn_2.append(f2)

pos_features,neg_features = [], []
"""This loop goes through each order id and creates a list of tuples where the tuples are (OrdID,coefficient,feature name)
   Append this to both the positive and negative features list
   Then create a new pandas dataframe from this list of tuples"""
for oid in df['OrdId']:
    pos_features.append(list(zip(itertools.repeat(oid,len(coef_2)),np.around(coef_2,4),fn_2)))
    neg_features.append(list(zip(itertools.repeat(oid,len(coef_1)),np.around(coef_1,4),fn_1)))
    
p = pd.DataFrame()
n = pd.DataFrame()
t1 = pd.DataFrame()
t2 = pd.DataFrame()
for i in range(len(pos_features)):
    n = n.from_records(neg_features[i],columns = ['OrdId','Coefficient','Keyword'])
    p = p.from_records(pos_features[i],columns = ['OrdId','Coefficient','Keyword'])
    t1 = t1.append(p,ignore_index = True)
    t2 = t2.append(n,ignore_index = True)


"""Pass each row to the find_counts method and create a long list of counts of keywords for each row.
   Then add this as a new column to our output dataframe"""
s_time = time.time()
counts_pos,counts_neg = [],[]
for i in range(len(t1)):
    counts_pos.append(find_counts(t1.loc[i]))
    counts_neg.append(find_counts(t2.loc[i]))
f_time = time.time()

print('Completed in', (f_time-s_time))

t1['Counts'] = pd.Series(counts_pos,index = t1.index)
t1['Score'] = t1['Coefficient'] * t1['Counts']

t2['Counts'] = pd.Series(counts_neg,index = t2.index)
t2['Score'] = t2['Coefficient'] * t2['Counts']


"""Save as excel file"""
# t1.to_excel(r'C:\Users\Natarajan\Documents\Jupyter notebooks\Sears\Results\Results 062317\{}{}_pos_chat_scores.xlsx'.format(file_name,ngram),index = None)
# t2.to_excel(r'C:\Users\Natarajan\Documents\Jupyter notebooks\Sears\Results\Results 062317\{}{}_neg_chat_scores.xlsx'.format(file_name,ngram),index = None)


"""________Metrics________"""

"""Set the type of classifier 
   logreg,SVM,SVM_SGD or Perceptron"""
classifier_type = 'logreg'
y_pred_class = None
if classifier_type is 'logreg':
    y_pred_class = y_pred_class_logreg
if classifier_type is 'SVM':
    y_pred_class = y_pred_class_svm
if classifier_type is 'SVM_SGD':
    y_pred_class = y_pred_class_sgd
if classifier_type is 'perceptron':
    y_pred_class = y_pred_class_ptrn    

confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1, 1] # True positive
TN = confusion[0, 0] # True Negative
FP = confusion[0, 1] # False positive
FN = confusion[1, 0] # False Negative

print(metrics.confusion_matrix(y_test, y_pred_class)) # The actual confusion matrix

"""Accuracy of the classifier"""
print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, y_pred_class))

"""Measure of how often the classifier is wrong"""
print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, y_pred_class))

"""Sensitivity: When the actual value is positive, how often is the prediction correct?"""
print(TP / float(TP + FN))
print(metrics.recall_score(y_test, y_pred_class))

"""Specificity: When the actual value is negative, how often is the prediction correct?"""
print(TN / float(TN + FP))

"""False Positive Rate: When the actual value is negative, how often is the prediction incorrect?"""
print(FP / float(TN + FP))

"""Precision: When a positive value is predicted, how often is the prediction correct?"""
print(TP / float(TP + FP))
print(metrics.precision_score(y_test, y_pred_class))

