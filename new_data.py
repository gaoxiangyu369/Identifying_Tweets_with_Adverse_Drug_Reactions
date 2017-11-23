from feature_selection import arff_to_pandas,drug_lexicon,find_feature

df = arff_to_pandas('train.arff')
concepts, tokens, drugList = drug_lexicon('ADR_lexicon.txt','drugs.txt')
find_feature('train.txt',df,drugList,tokens,concepts)
df.to_csv('train.csv')

df1 = arff_to_pandas('dev.arff')
concepts, tokens, drugList = drug_lexicon('ADR_lexicon.txt','drugs.txt')
find_feature('dev.txt',df1,drugList,tokens,concepts)
df1.to_csv('dev.csv')

df3 = arff_to_pandas('train.arff')
concepts, tokens, drugList = drug_lexicon('ADR_lexicon.txt','drugs.txt')
find_feature('train.txt',df3,drugList,tokens,concepts)
df3.to_csv('test.csv')