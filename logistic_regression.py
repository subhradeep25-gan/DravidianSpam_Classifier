import pandas as pd
import numpy as np
mal_data = pd.read_csv("https://github.com/subhradeep25-gan/DravidianSpam_Classifier/blob/master/Malayalam_offensive_data_Training-YT.csv")

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

def lower_X(text):
    if(type(text)==str):
        s = ''
        for i in text:
            try:
                s = s+ i.lower()
            except:
                s = s +i
        return s
    else:
        return 'None'

mal_data["Tweets"] = mal_data["Tweets"].map(lower_X)

from sklearn.model_selection import train_test_split

X_train_sentence, X_test_sentence, y_train, y_test = train_test_split(mal_data["Tweets"], mal_data["Labels"], train_size = 0.8, random_state = 200)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(X_train_sentence)

X_train = vectorizer.transform(X_train_sentence)
X_test = vectorizer.transform(X_test_sentence)

from sklearn.linear_model import LogisticRegression

lorReg = LogisticRegression()

lorReg.fit(X_train, y_train)
predictions = lorReg.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
    'max_iter' : [100, 1000, 2500, 5000]
    }
]

clf = GridSearchCV(lorReg, param_grid = param_grid, cv = 3, verbose = True, n_jobs = -1)

best_clf = clf.fit(X_test,y_test)

print(best_clf.best_params_)
print(f'Test Accuracy - : {best_clf.score(X_test, y_test):.3f}')
lr = LogisticRegression(C=0.9, max_iter = 1000, penalty = 'l2')
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))