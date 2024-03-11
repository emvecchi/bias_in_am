import json, sys

def readAnnotations(annsFile):
    annotations = { }

    for line in open(annsFile, "r"):
        cmvId, ann = line.split('\t')
        annotations[cmvId.strip()] = ann.strip()
        
    return annotations

def readAuthorsAnns(dataFile, anns):
    authorsAnns = { }

    with open(dataFile, 'r') as data:
        for line in data:
            cmv = json.loads(line)
            cmvId = cmv["id"]

            if "author" in cmv:
                author = cmv["author"]

                if cmvId in anns:
                    if author not in authorsAnns:
                        authorsAnns[author] = [ anns[cmvId] ]
                    else:
                        authorsAnns[author].append(anns[cmvId])

            for comment in cmv["comments"]:
                cmvId = comment["id"]

                if "author" not in comment or cmvId not in anns:
                    continue

                author = comment["author"]
                if author not in authorsAnns:
                    authorsAnns[author] = [ anns[cmvId] ]
                else:
                    authorsAnns[author].append(anns[cmvId])
    
    result = { }
    moreThanOne = 0
    for author, anns in authorsAnns.items():
        anns = list(set(anns))
        if len(anns) > 1:
            moreThanOne += 1
        else:
            result[author] = anns[0]

    print("Number of authors with more than one gender: ", moreThanOne)
    return result

if __name__ == '__main__':
    args = sys.argv[1:]
    
    inFile = args[0]
    annotations = readAnnotations(args[1])
    authors = readAuthorsAnns(inFile, annotations)

    outFile = open(args[2], "w")
    with open(inFile, 'r') as data:
        for line in data:
            cmv = json.loads(line)
            cmvId = cmv["id"]

            if "author" in cmv and cmv["author"] in authors:
                outFile.write(cmvId + "\t" + authors[cmv["author"]] + "\n")
                
            for comment in cmv["comments"]:
                cmvId = comment["id"]

                if "author" not in comment:
                    continue

                author = comment["author"]
                if author not in authors:
                    continue

                outFile.write(cmvId + "\t" + authors[comment["author"]] + "\n")

    outFile.close()