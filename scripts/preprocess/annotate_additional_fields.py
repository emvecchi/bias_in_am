import json, bz2
import os, sys

def readAnnFile(inFile):
    result = { }
    for line in open(inFile):
        annId, annValue = line.strip().split("\t")
        if annId not in result:
            result[annId] = [ ]

        result[annId].append(annValue)

    return result

def readAnnotations(annDir):
    annotations = { "threads" : { }, "replies" : { }, "authors" : { }}

    for filename in os.listdir(annDir):
        anns = readAnnFile(annDir + "/" + filename )
        annType, annField, _ = filename.split(".")
        annotations[annField][annType] = anns

    return annotations

def annotateCMV(cmv, anns, fieldName):
    for annType, values in anns.items():
        cmvAnn = None

        if fieldName not in cmv:
            print("No field", cmv, fieldName)
            pass
        else:
            cmvID = cmv[fieldName]
    
            if cmvID not in values:
                #print("No", annType, "for", cmvID)
                pass
            else:
                cmvAnns = set(values[cmvID])
                if len(cmvAnns) > 1:
                    #print("Too many options:", cmvAnns, cmvID)
                    pass
                else:
                    cmvAnn = list(cmvAnns)[0]

        if cmvAnn is None:
            cmvAnn = "UNK"

        cmv[annType] = cmvAnn

if __name__ == '__main__':
    args = sys.argv[1:]
    
    inFile = bz2.open(args[0], 'rb')
    annotations = readAnnotations(args[1])
    outFile = open(args[2], "w")
    
    for line in inFile:
        cmv = json.loads(line)
        annotateCMV(cmv, annotations["threads"], "id")
        annotateCMV(cmv, annotations["authors"], "author")

        for comment in cmv["comments"]:
            annotateCMV(comment, annotations["replies"], "id")
            annotateCMV(comment, annotations["authors"], "author")

        json.dump(cmv, outFile)
        outFile.write("\n")
        outFile.flush()

        