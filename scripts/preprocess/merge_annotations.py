import json, sys

def readAnnotations(annsFile):
    annotations = { }

    for line in open(annsFile, "r"):
        cmvId, ann = line.split('\t')
        annotations[cmvId.strip()] = ann.strip()
        
    return annotations


if __name__ == '__main__':
    args = sys.argv[1:]
    
    inFile = args[0]
    explicit = readAnnotations(args[1])
    implicit = readAnnotations(args[2])
    topics = readAnnotations(args[3])
    out = args[4]

    outFile = open(args[4], "w")
    with open(inFile, 'r') as data:
        for line in data:
            cmv = json.loads(line)
            cmvId = cmv["id"]

            cmv["explicit_gender"] = explicit[cmvId] if cmvId in explicit else "UNK"
            cmv["author_gender"] = implicit[cmvId] if cmvId in implicit else "UNK"
            cmv["topic"] = topics[cmvId]
                
            for comment in cmv["comments"]:
                cmvId = comment["id"]

                comment["explicit_gender"] = explicit[cmvId] if cmvId in explicit else "UNK"
                comment["author_gender"] = implicit[cmvId] if cmvId in implicit else "UNK"
            
            json.dump(cmv, outFile)
            outFile.write("\n")
    outFile.close()