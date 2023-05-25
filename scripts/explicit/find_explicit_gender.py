import json, bz2
import spacy
import string

NLP = spacy.load("en_core_web_sm")
SEP = "##"

def clearText(text):
    result = [ ]
    for word in text.lower().split():
        word = "".join([c for c in word if c in string.ascii_lowercase])
        result.append(word)

    result = " ".join(result)
    return " ".join(result.split())
        
def filterGenderSentences(inFile, genders, out):
    
    for line in inFile:
        cmv = json.loads(line)
    
        for text in cmv["selftext"].split("\n"):
            doc = NLP(text)
            for sentence in doc.sents:
                lemmas = [ token.lemma_ for token in sentence ]
                if "I" not in lemmas:
                    continue

                text = str(sentence)
                allText = "##" + "##".join(text.lower().split() + clearText(text).split()) + "##"
                    
                gSentence = False
                foundG = None
                for g in genders:
                    if "##" + g + "##" in allText:
                        foundG = g
                        gSentence = True
                        break

                if not gSentence:
                    for lemma in lemmas:
                        if lemma in genders:
                            foundG = lemma
                            gSentence = True
                            break
                    
                if gSentence:
                    out.write(str(sentence) + "\n")
                    out.flush()
            

def readGenders(filename):
    genders = set([])
    for line in open(filename, 'r'):
        # clear white spaces
        words = line.strip().split()
        
        genders.add(" ".join(words))
        genders.add("".join(words))
        genders.add(SEP.join(words))

        clearWords = clearText(line).split()

        genders.add(" ".join(clearWords))
        genders.add("".join(clearWords))
        genders.add(SEP.join(clearWords))

    return genders

if __name__ == '__main__':
    import os, sys

    args = sys.argv[1:]
    inFile = bz2.open(args[0], 'rb')

    genders = readGenders(args[1])
    out = open(args[2], 'w')
    filterGenderSentences(inFile, genders, out)