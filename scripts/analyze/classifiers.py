import numpy as np
import torch
import pandas as pd
import tqdm
import argparse
from data import PredictionDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import torch.nn.functional as F

""" 

Twitter-roBERTa-base for Sentiment Analysis (cardiffnlp/twitter-roberta-base-sentiment-latest)
		Updated version (2021)

Input: List of texts
Output: Returns a list of [positve, neutral, negative] scores for each text

"""


def sentiment_classifier(dataset, text_col):
    from transformers import AutoModelForSequenceClassification
    from transformers import TFAutoModelForSequenceClassification
    from transformers import AutoTokenizer, AutoConfig
    from scipy.special import softmax

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




"""

Emotion English DistilRoBERTa-base (j-hartmann/emotion-english-distilroberta-base)

Input: List of text for classification
Output: Returns a dataframe, each row is a text from the list, columns include top scoring emotion
		and then all 7 emotions with their scores
"""


def emotion_classifier(text_list):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
    # Create class for data preparation
    class SimpleDataset:
        def __init__(self, tokenized_texts):
            self.tokenized_texts = tokenized_texts

        def __len__(self):
            return len(self.tokenized_texts["input_ids"])

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.tokenized_texts.items()}

    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)
    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer(text_list, truncation=True, padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)
    # Run predictions
    predictions = trainer.predict(pred_dataset)
    # Transform predictions to labels
    preds = predictions.predictions.argmax(-1)
    labels = pd.Series(preds).map(model.config.id2label)
    scores = (np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True)).max(1)
    # scores raw
    temp = (np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True))
    # container
    anger = []
    disgust = []
    fear = []
    joy = []
    neutral = []
    sadness = []
    surprise = []
    # extract scores (as many entries as exist in text_list)
    for i in range(len(text_list)):
        anger.append(temp[i][0])
        disgust.append(temp[i][1])
        fear.append(temp[i][2])
        joy.append(temp[i][3])
        neutral.append(temp[i][4])
        sadness.append(temp[i][5])
        surprise.append(temp[i][6])
    # Create DataFrame with texts, predictions, labels, and scores
    df = pd.DataFrame(
        list(zip(text_list, preds, labels, scores, anger, disgust, fear, joy, neutral, sadness, surprise)),
        columns=['text', 'pred', 'label', 'score', 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])
    return (df)


"""


Toxicity Classification Model (SkolkovoInstitute/roberta_toxicity_classifier)

Input: List of texts
Output: Returns a list of probability scores for each cateogry [neutral, toxic]

"""


def toxicity_classifier(df, args):
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    # load tokenizer and model weights
    tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    dataset = PredictionDataset(dataset=df, tokenizer=tokenizer, text_col=args.text_col)
    trainer = Trainer(model=model, tokenizer=tokenizer)

    ouput = trainer.predict(dataset)
    predictions = F.softmax(torch.tensor(ouput.predictions), dim=-1).tolist()
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path to data to generate predictions for')
    parser.add_argument('--text_col', type=str, help='name of the text column in the data')
    parser.add_argument('--output_path', type=str, help='path to save the predictions')
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep='\t')
    # replace nan with empty string in text column
    df[args.text_col] = df[args.text_col].fillna('')
    text_list = df[args.text_col].tolist()
    print('Running Sentiment Analysis')
    predicted_labels, sentiment_scores = sentiment_classifier(df, args.text_col)
    # each element in this list is a dictionary with keys 'positive', 'neutral', 'negative'. add each value to a column to the dataframe
    df['sentiment_positive'] = [x[0] for x in sentiment_scores]
    df['sentiment_neutral'] = [x[1] for x in sentiment_scores]
    df['sentiment_negative'] = [x[2] for x in sentiment_scores]
    df["sentiment"] = predicted_labels
    # save dataframe
    df.to_csv(args.output_path, sep='\t', index=False)
    """
    print('Running Emotion Analysis')
    emotion_scores = emotion_classifier(text_list)
    """
    print('Running Toxicity Analysis')
    toxicity_scores = toxicity_classifier(df, args)
    """
    # emotion_scores is a dataframe, just add the new columns to df
    df['emotion_anger'] = emotion_scores['anger']
    df['emotion_disgust'] = emotion_scores['disgust']
    df['emotion_fear'] = emotion_scores['fear']
    df['emotion_joy'] = emotion_scores['joy']
    df['emotion_neutral'] = emotion_scores['neutral']
    df['emotion_sadness'] = emotion_scores['sadness']
    df['emotion_surprise'] = emotion_scores['surprise']
    """
    # toxicity_scores is a list of lists, add each list to a column in df
    df['toxicity_neutral'] = [x[0] for x in toxicity_scores]
    df['toxicity_toxic'] = [x[1] for x in toxicity_scores]
    df.to_csv(args.output_path, sep='\t', index=False)
