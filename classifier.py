import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv('train.csv')
dev = pd.read_csv('dev.csv')

train['drug_mention'] = LabelEncoder().fit_transform(train['drug_mention'])
train['ADR_mention'] = LabelEncoder().fit_transform(train['ADR_mention'])
train['mood'] = LabelEncoder().fit_transform(train['mood'])

dev['drug_mention'] = LabelEncoder().fit_transform(dev['drug_mention'])
dev['ADR_mention'] = LabelEncoder().fit_transform(dev['ADR_mention'])
dev['mood'] = LabelEncoder().fit_transform(dev['mood'])

train_data = train.values[:, 1:--1]
train_label = train.values[:, -1]

dev_data = dev.values[:, 1:-1]
dev_label = dev.values[:, -1]

clf = GaussianNB()
clf = clf.fit(train_data, train_label)
dev_pred = clf.predict(dev_data)

print(classification_report(dev_label, dev_pred, target_names=['N', 'Y']))
print(confusion_matrix(dev_label, dev_pred, labels=['Y', 'N']))