def get_tan_word_features(df):
    # This function adds the missing word features used in Tan et al (2016) to the feature df 
    import re
    import nltk
    import math
    df['tmp_text'] = df['text'].str.replace(r'\*Hello, users of CMV!.*', '', regex=True)
    #
    # definite & indefinite articles
    definite_articles = ['the']
    indefinite_articles = ['a','an']
    df['definite_article_count'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(definite_articles))) 
    df['definite_article_perc'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(definite_articles))) / df['nwords']
    df['indefinite_article_count'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(indefinite_articles)))
    df['indefinite_article_perc'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(indefinite_articles))) / df['nwords']
    #
    # second person pronoun
    second_person = ['you', 'your', 'yours', 'yourself', 'yourselves']
    df['second_person'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(second_person))) / df['nwords']
    #
    # first person plural
    first_person_pl = ['we', 'us', 'our', 'ours', 'ourselves']
    df['first_person_pl'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(first_person_pl))) / df['nwords']
    #
    # hedging
    hedges = ['maybe', 'perhaps', 'possibly', 'potentially', 'likely', 'probably', 'could', 'might', 'may', 'can', 'should']
    df['hedge_words'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(hedges)))
    df['hedge_words_perc'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(hedges))) / df['nwords']
    #
    # .com|.edu|.org links
    df['com_link_count'] = df['tmp_text'].str.lower().str.count(r'\b[\w.-]+\.com\b')
    df['edu_link_count'] = df['tmp_text'].str.lower().str.count(r'\b[\w.-]+\.edu\b')
    df['org_link_count'] = df['tmp_text'].str.lower().str.count(r'\b[\w.-]+\.org\b')
    df['pdf_link_count'] = df['tmp_text'].str.lower().str.count(r'\b[\w.-]+\.pdf\b')
    #
    # punctuation
    df['num_quotations'] = df['tmp_text'].str.count(r'["\']')
    df['num_question'] = df['tmp_text'].str.count(r'\?')
    df['num_exclamation'] = df['tmp_text'].str.count(r'\!')
    #
    #  sentence boundaries
    nltk.download('punkt')
    df['sentence_count'] = df['tmp_text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    df['paragraph_count'] = df['tmp_text'].str.split(r'\n{2,}').apply(len)
    #
    # word entropy
    df['tokenized_text'] = df['tmp_text'].apply(nltk.word_tokenize)
    df['word_frequencies'] = df['tokenized_text'].apply(nltk.FreqDist)
    df['total_words'] = df['tokenized_text'].apply(len)
    df['word_probabilities'] = df['word_frequencies'].apply(lambda x: {word: freq / sum(x.values()) for word, freq in x.items()})
    df['entropy'] = df['word_probabilities'].apply(lambda x: -sum(prob * math.log2(prob) for prob in x.values()))
    df = df.drop(columns=['word_frequencies','total_words','word_probabilities','tokenized_text'])
    #
    # examples
    df['example_count'] = df['tmp_text'].str.count(r'example', flags=re.IGNORECASE)
    #
    df = df.drop(columns=['tmp_text'])
    return(df)

"""
Toxicity Classification Model (SkolkovoInstitute/roberta_toxicity_classifier)

Input: Dataframe, text_col
Output: Returns a list of probability scores for each cateogry [neutral, toxic] for each text in text_col

"""
def toxicity_classifier(df, text_col):
    import pandas as pd
    import numpy as np
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
    import torch.nn.functional as F
    from data import PredictionDataset
    # load tokenizer and model weights
    tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    dataset = PredictionDataset(dataset=df, tokenizer=tokenizer, text_col=text_col)
    trainer = Trainer(model=model, tokenizer=tokenizer)
    ouput = trainer.predict(dataset)
    predictions = F.softmax(torch.tensor(ouput.predictions), dim=-1).tolist()
    return(predictions)

""" 
Twitter-roBERTa-base for Sentiment Analysis (cardiffnlp/twitter-roberta-base-sentiment-latest)
        Updated version (2021)

Input: Dataframe, Text Column
Output: Returns a list of [positve, neutral, negative] scores for each text in text_col

"""
def sentiment_classifier(dataset, text_col):
    from transformers import AutoModelForSequenceClassification
    from transformers import TFAutoModelForSequenceClassification
    from transformers import AutoTokenizer, AutoConfig, Trainer
    from scipy.special import softmax
    import pandas as pd
    import numpy as np
    from data import PredictionDataset
    import torch
    import torch.nn.functional as F
    # Preprocess text (username and link placeholders)
    # NOTE: this won't be that important for our datasets (non twitter)
    # TO-DO: do check it doesn't have neg impact
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # set max sequence length of tokenizer
    tokenizer.max_len = 512
    tokenizer.model_max_length = 512
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    dataset = PredictionDataset(dataset=dataset, tokenizer=tokenizer, text_col=text_col)
    trainer = Trainer(model=model, tokenizer=tokenizer)
    id2label = config.id2label
    print(id2label)
    ouput = trainer.predict(dataset)
    predictions = F.softmax(torch.tensor(ouput.predictions), dim=-1).tolist()
    #
    # generate the predicted label and add it to the dataframe
    predicted_labels = np.argmax(ouput.predictions, axis=1).flatten()
    predicted_labels = [id2label[pred] for pred in predicted_labels]
    return predicted_labels, predictions

