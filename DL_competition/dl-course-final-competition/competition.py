import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_data(data_path: str):

    origin_data = pd.read_csv(data_path)

    return origin_data


def test():
    function = dfdaf


def function(self, parameter_list):
    print()
    anandfdf

    function


def split_keyword(words):
    if pd.isna(words):
        return []
    else:
        return words.split(',')


def remove_stopword(data_list):
    new_data = []
    for i, sentence in enumerate(data_list):
        for word in sentence:
            if word in stop_word:
                sentence.remove(word)
        data_list.iloc[i] = sentence


train_data = load_data("train_data.csv")
test_data = load_data("test_data.csv")

with open("stop_words.csv", 'r') as fp:
    raw_data = fp.readlines()
    stop_word = set([word.strip() for word in raw_data])

# keyword
keyword_list_train = train_data['keyword'].apply(split_keyword)
keyword_list_test = test_data['keyword'].apply(split_keyword)

# 切title
seg_list_train = train_data['title'].apply(lambda x: list(jieba.cut(x)))
seg_list_test = test_data['title'].apply(lambda x: list(jieba.cut(x)))

# remove stopword
remove_stopword(seg_list_train)
remove_stopword(seg_list_test)

# TF-IDF
sentence_vectorizer = TfidfVectorizer()
title_list = [' '.join(words) for words in seg_list_train]
X_old = sentence_vectorizer.fit_transform(title_list)

title_list_test = [' '.join(words) for words in seg_list_test]
X_test_old = sentence_vectorizer.transform(title_list_test)

keyword_vectorizer = TfidfVectorizer()
keyword_list_train_new = [' '.join(words) for words in keyword_list_train]
keyword_train = keyword_vectorizer.fit_transform(keyword_list_train_new)

keyword_list_test_new = [' '.join(words) for words in keyword_list_test]
keyword_test = keyword_vectorizer.transform(keyword_list_test_new)

# stack X & keyword => feature
X = hstack([X_old, keyword_train])
X_test = hstack([X_test_old, keyword_test])

# 降 維
X_reduce = TruncatedSVD(n_components=300).fit_transform(X)
X_reduce.shape

y = train_data['label'].tolist()

print("start training")

# SVM
clf = make_pipeline(StandardScaler(with_mean=False),
                    SVC(gamma='auto', verbose=True))
clf.fit(X_reduce, y)

predicts = clf.predict(X_test)


with open('submission.csv', 'w') as fp:
    fp.write('id,lable\n')
    for i, predict in enumerate(predicts):
        fp.write('%d,%d' % (i, predict))
