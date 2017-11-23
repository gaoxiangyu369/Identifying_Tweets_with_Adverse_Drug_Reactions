# Identifying Tweets with Adverse Drug Reactions

The goal of this Project is to assess the effectiveness of some (supervised) Machine Learning methods on the problem of determining whether a tweet contains an ADR.

The data involves raw tweets separated into three sub-directories – train, dev, and test – each dataset is represented by a reduced vector space model (VSM).  The train and dev data are labeled with Y or N, indicating whether the tweet depicts an ADR or not respectively.

## Feature Selection

### Methodology

Identifying potential tweets with ADRs requires the mentions of both drug(s) and medical events.

Furthermore, when people happen to encounter the problems especially some health-related aspects, they usually express their emotion in a negative way.

### Preprocessing

By removing hashtags (#), URLs and at (@), a corpus of cleaner tweets is generated. In addition, each tweet is tokenized into a bunch of single words.

### Word Features

Given labeled training tweets, **Mutual Information** measures the dependency between each term and the category. Higher values mean higher dependency.

### ADR Lexicon Feature

This binary feature shows whether or not the current tweet has some related content in the ADR lexicon. A regular expression defined as below is used to find as many perfect matches as possible:

```
'(^|\s+|<PUNCTUATION>)'<ADR LEXICON>'($|\s+|<PUNCTUATION>)'
```

### Drug Lexicon Feature

Each tweet is compared with the drugs to indicates whether or not the current word in the drug lexicon.

### Negation Feature

This binary feature describes whether or not the current tweet is negative. The negation is identified based on sentiment analyzers from the TextBlob.

## Results Analysis

A Naïve Bayes classifier trained on word, ADR lexicon, Drug lexicon, negation and Drug-Syndrome frequency features is present to show the overall performance on task of identifying the potential tweets with ADRs detection. Compared with the baseline model with initial features, a model with more proper features highly improves the prediction on depictions of ADRs in tweets.

## Code Documentation

### ADR_lexicon.txt

It includes 13,591 frequently observed phrases in user posts in social media which were annotated as ADRs from [COSTART, SIDER and a subset of CHV](http://diego.asu.edu/downloads/publications/ADRMine/ADR_lexicon.tsv).

### drugs.txt

The 334 drugs represent the 81 drugs present in ADR tweets from the Nikfarjam 2015 study, the New York State Department of Health top 150 drugs list, and Chemical and Engineering News' 2014 top 50 drugs list.

For more details about the data, please refer to ericbenz's homepage about [TwitterDrugs](https://github.com/ericbenz/TwitterDrugs).

### feature_selection.py

Given raw tweets, drug list and ADR lexicon, this program generates 4 new features including ADR_mention, drug_mention, mood and drug_syndrome. The .py file contains 3 functions for outer invoke.

arff_to_pandas is used to transform a .arff file to pandas format in python. drug_lexicon is used to extract drug and ADR terms from the source file. find_feature is used to find features mentioned before and compose them into a new data frame.

### new_data.py

After feature selection, new features plus the original 92 terms will form a vector space model for each tweets and export as .csv file for train, dev, and test.

### classifier.py

This file is used to train and build a Naive Bayes model base on training data. After that, the developing data is used to evaluate the model's performance by print the classification report and the confusion matrix.

