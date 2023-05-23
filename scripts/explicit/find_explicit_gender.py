import json, bz2
#import spacy

def filterIDs(inFile, idsFile, outFile):
    threadIds = set([ line.split('"')[1] for line in open(idsFile, "r") ])
    
    found = 0
    for line in inFile:
        cmv = json.loads(line)

        if cmv["id"] in threadIds:
            json.dump(cmv, outFile)
            outFile.write("\n")
            found += 1
            outFile.flush()
            
    print("Found threads", found)
            
if __name__ == '__main__':
    import os, sys

    args = sys.argv[1:]
    inFile = bz2.open(args[0], 'rb')
    genders = set([ line.strip() for line in open(args[1], 'r') ])

    for line in inFile:
        cmv = json.loads(line)
    
        print("TITLE", cmv["title"])
        text = cmv["selftext"]

        #if "&gt; *Hello," in text:
        #    parts = text.split("&gt; *Hello,")
        #    text = parts[0]
            
        doc = nlp(text)
        for sentence in doc.sents:
            lemmas = [ token.lemma_ for token in sentence ]
            if "I" not in lemmas or "be" not in lemmas:
                continue

            kids = { }
            beToks = [ ]
            for tok in sentence:
                if tok.head not in kids:
                    kids[tok.head] = [ ]

                kids[tok.head].append(tok)

                if tok.lemma_ in [ "be" ]:
                    beToks.append(tok)

            for tok in beToks:
                if tok in kids:
                    if all( [ t.lemma_ != "I" for t in kids[tok] ]):
                        continue

                    for t in kids[tok]:
                        if t.lemma_ not in demos:
                            continue

                        toPrint = [ ]
                        if t in kids:
                            toPrint += kids[t]

                        print(t, toPrint, " ".join(str(sentence).strip().split()))

            #    iamparts = sentence.split("I am a ")
            #    print(" ".join(iamparts[1].split()[:20]))

        if nr < 0:
            break
        nr -= 1


    inFile.close()  