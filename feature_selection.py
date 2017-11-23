import re
from arff2pandas import a2p
from textblob import TextBlob as tb
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.corpus import stopwords

def arff_to_pandas(filename):
    with open(filename) as f:
        df = a2p.load(f)
    df.columns = [re.sub(r'@(NUMERIC|{N,Y})', '', col) for col in df.columns]

    return df

def drug_lexicon(file1, file2):
    with open(file1, 'r') as content:
        lines = content.readlines()

    with open(file2, 'r') as content:
        drugs = content.readlines()

    concepts = []
    for line in lines:
        concepts.append(line.strip())

    stop_words = set(stopwords.words('english'))
    tokens = []
    for concept in concepts:
        tokens.extend([i for i in concept.lower().split() if i not in stop_words])

    tokens = list(set(tokens))

    drugList = []
    for line in drugs:
        drugList.append(line.strip('\n'))

    return concepts, tokens, drugList


def find_feature(filename, dataFrame, drugList, tokens, concepts):
    with open(filename, 'r', encoding='ascii', errors='ignore') as content:
        rawData = content.readlines()

    corpus = []
    for line in rawData:
        account, category, tweet = line.strip().split('\t')
        tweet = re.sub(r'(@\_*\w+\_*)|(http://.+\s?)|(pic.twitter.com/.+\s?)|(\#)', '', tweet)
        corpus.append(tweet.lower().strip())

    contents = []
    for line in corpus:
        token = re.findall(r'[A-Za-z]+', line)
        contents.append(token)

    lexicon = []
    for content in contents:
        count = 0
        for token in content:
            if token in tokens:
                count += 1
        lexicon.append(count)

    drugFeature = []
    for content in contents:
        count = 0
        for token in content:
            if token in drugList:
                count += 1
        drugFeature.append(count)

    drug_syndrome = list(map(lambda x: x[0] + x[1], zip(lexicon, drugFeature)))

    drug = []
    for i in drugFeature:
        if i > 0:
            drug_mention = 'Y'
        else:
            drug_mention = 'N'
        drug.append(drug_mention)

    head = '(^|\s+|\'|!|\"|\$|&|\(|\)|\*|\+|,|\-|\.|/|:|;|=|>|\?|\[|\]|\_|~)'
    tail = '($|\s+|\'|!|\"|\$|&|\(|\)|\*|\+|,|\-|\.|/|:|;|=|>|\?|\[|\]|\_|~)'
    matchObj = []

    sentiment = []
    for tweet in corpus:
        count = 0
        blob = tb(tweet, analyzer=NaiveBayesAnalyzer())
        if blob.sentiment[0] == 'neg':
            mood = 'Y'
        else:
            mood = 'N'
        for event in concepts:
            pattern = head + event.strip() + tail
            match = re.search(pattern, tweet.strip())
            if match:
                count += 1
        matchObj.append(count)
        sentiment.append(mood)

    adr = []
    for item in matchObj:
        if item > 0:
            mention = 'Y'
        else:
            mention = 'N'
        adr.append(mention)

    col_name = list(dataFrame.columns)
    dataFrame.insert(col_name.index('class'), 'drug_syndrome', drug_syndrome)
    dataFrame.insert(col_name.index('class'), 'drug_mention', drug)
    dataFrame.insert(col_name.index('class'), 'mood', sentiment)
    dataFrame.insert(col_name.index('class'), 'ADR_mention', adr)


df = arff_to_pandas('train.arff')
concepts, tokens, drugList = drug_lexicon('ADR_lexicon.txt','drugs.txt')
find_feature('train.txt',df,drugList,tokens,concepts)
df.to_csv('train.csv')


