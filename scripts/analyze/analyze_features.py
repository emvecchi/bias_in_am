def readFeatureTypes(dirName):
    from os import listdir
    from collections import defaultdict

    result = defaultdict(lambda : "unknown")
    for fileName in listdir(dirName):
        traitType = fileName.split(".")[0]
        for line in open(dirName + "/" + fileName, "r"):
            trait = line.strip()
            if trait in result:
                if traitType == "opinionated" and result[trait] == "personality":
                    pass # all is good
                else:
                    print("Train with the same type:", trait, result[trait], traitType)
                    continue

            result[trait] = traitType

    return result
        
def clearFeature(feature):
    return " ".join(feature.strip().lower().replace(".", " ").split())


def tokenizeTexts(texts, model_ckpt = "distilbert-base-uncased"):
    from transformers import AutoTokenizer
    from collections import defaultdict
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    result = [ ]
    for text in texts:
        result.append(tokenizer(text, return_tensors="pt", padding=True, truncation=True))
        
    return result

def generateHiddenStates(allTokens, model_ckpt = "distilbert-base-uncased"):
    import torch
    from transformers import AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_ckpt).to(device)

    result = [ ]
    for tokens in allTokens:
        with torch.no_grad():
            inputs = {k:v.to(device) for k,v in tokens.items()}
            last_hidden_state = model(**inputs).last_hidden_state
            cls_token = last_hidden_state[:,0][0].numpy()
            result.append(cls_token) # -> NumPy array

    return result

def generatePointsToPlot(texts):
    import numpy as np 
    import umap

    tokens = tokenizeTexts(texts)
    data = generateHiddenStates(tokens)        
    mapper = umap.UMAP(n_components=2, metric="cosine").fit(np.array(data))
    X = mapper.embedding_[:, 0]
    Y = mapper.embedding_[:, 1]   
    return X, Y


if __name__ == '__main__':
    import csv, collections, sys

    args = sys.argv[1:]

    featureTypes = readFeatureTypes(args[0])
    sys.exit()
    features = collections.defaultdict(int)
    categories = { }
    with open(args[0], 'r') as csvfile:
        # header line
        csvfile.readline()

        for row in csv.reader(csvfile):
            # timestamp agree ID lang
            row = row[4:]
            assert len(row) == 50

            for word in row:
                word = word.replace(".", " ")
                word = word.lower().strip()

                if word not in featureTypes:
                    print(word)

                features[word] += 1

                categories[word] = traits[word]
                
                if word in opinions:
                    print(word)
                
                if word in opiniated:
                    print(word)
                    

    print("Number of features", len(features))
    features = [ (freq, word) for (word, freq) in features.items() ]
    features.sort(reverse=False)
    #for (freq, word) in features:
    #    if freq == 1:
    #        print(word)

    #for word, wordType in knownFeatures.items():
    #    print(word, wordType)