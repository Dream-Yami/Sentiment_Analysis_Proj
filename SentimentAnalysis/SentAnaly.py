import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
import nltk
#nltk.download() only do

# Imported the review data frame here
# messing around with using the dataframe
df = pd.read_csv('Reviews\Reviews.csv')
#print(df.shape) for all
#print(df['Text'].values[0])
df = df.head(500)

#ordering the score vals into a bar graph
ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()

#NLTK testing__________________________
example = df['Text'][50]
print(example)

#Utilizes NLTK to parses the given string into words and tags them with their part of speech.
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
print(tagged[:10])

#Chunnk the tokens
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
#END NLTK PRACTICE


#VADER SENTIMENT ANALYSIS (VALENCE AWARE DICTIONARY sEntiment REASONER)
#NLTK's sentimentIntensityAnalyzer will provide scores neg,neu,pis to each word
#NOTE DOES NOT ACCOUNT FOR RELATIONSHIP BETWEEN WORDS

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia = SentimentIntensityAnalyzer()

#test on one sentence
print(sia.polarity_scores('I am a happy boi'))
#Running a loop to test on the entire CSV file
res = {}
#Res is a dictionary of myID with keys holding polaroity scores of the matching ID
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)
    
print(res)
print(pd.DataFrame(res).T)
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
print(vaders.head())

#BarPlots vader results
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

#-------------------------
#Roberta Pretained Model
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

#First time use will take a bit to install the weights
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
print(example)

#Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)

#Running the Roberta Model on the entire model
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    #Sometimes it breaks as it cannot handle reviews that are too long
    except RuntimeError:
        print(f'Broke for id {myid}')
        
#Combining Results Dictionary
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')