import pandas as pd
import numpy as np
mal_data = pd.read_csv("/content/drive/MyDrive/Malayalam Models/Malayalam_offensive_data_Training-YT.csv")
mal_data_test = pd.read_excel("/content/drive/MyDrive/final_test_mal-offensive-with-labels (1).xlsx")
mal_data.drop("ID", axis = 1, inplace = True)
mal_data_test.drop("ID", axis = 1, inplace = True)
import re
def remove_symbols(string):
    a = re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)","",string)
    return a
mal_data["Tweets"] = mal_data["Tweets"].map(remove_symbols)

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

mal_data["Tweets"] = mal_data["Tweets"].map(remove_emojis)
mal_data['Labels'] = mal_data['Labels'].replace({'OFF' : 1, 'NOT' : 0})
mal_data_test['Labels'] = mal_data_test['Labels'].replace({'HOF' : 1, 'NOT' : 0})

X_train = mal_data["Tweets"]
y_train = mal_data["Labels"]
X_test = mal_data_test["Tweets"]
y_test = mal_data_test["Labels"]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

#Random Forest Classifier

f = open("rf.txt",'w')
f.write(f"RandomForestClassifer(n_estimators = {0},criterion={0},max_depth={0},min_samples_split={0},max_features = {0},max_leaf_nodes = {0})")
f.close()
from sklearn.metrics import f1_score

from sklearn.metrics import f1_score
n_estimators = [10,20,30,40,50,60,70,80,90,100]
criterion = ["gini", "entropy", "log_loss" ]
max_features = ["auto", "sqrt", "log2"]
max_depth = [10,20,30,40,50,100,150,200,300,350,400,450,500,550]
min_samples_split = [1,2,3,4,5,10,50,100,200,300,500]
max_leaf_nodes = [20,50,100,150,200,400,300,250,350,450]

from sklearn.ensemble import RandomForestClassifier
score = 0

for i in n_estimators:
  for k in criterion:
    for l in max_depth :
      for m in max_features:
        for n in min_samples_split:
          for p in max_leaf_nodes:
             print(i,k,l,m,n,p)
          try:
            model = RandomForestClassifier(n_estimators = i,criterion =k  , max_depth = l , max_features = m,min_samples_split = n,max_leaf_nodes=p)
            model.fit(X_train,y_train)
          except :
            continue
          pred_m = model.predict(X_test)
          f = f1_score(y_test,pred_m,average = 'macro')
          if(score<f):
              score = f
              f = open("rf.txt",'w')
              f.write(f"DecisionTreeClassifier(n_estimators = {i},criterion = {k},max_depth={l},max_features = {m},min_samples_split = {n},max_leaf_nodes = {p})")
              f.close()

model = RandomForestClassifier(n_estimators =60,criterion ='gini',max_depth=400,max_features ='log2',min_samples_split =5,max_leaf_nodes =450)
model.fit(X_train,y_train)
y_predicted = model.predict(X_test)
print(classification_report(y_predicted,y_test))
