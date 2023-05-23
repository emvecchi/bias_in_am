import json, bz2

def filterIDs(inFile, threadIds, replyIds, outFile):    
    found = 0
    replyFound = 0
    for line in inFile:
        if found >= len(threadIds):
            break

        cmv = json.loads(line)

        if cmv["id"] not in threadIds:
            continue

        found += 1

        newComments = [ ]
        for comment in cmv["comments"]:
            if comment["id"] in replyIds:
                newComments.append(comment)
                replyFound += 1

        cmv["comments"] = newComments

        json.dump(cmv, outFile)
        outFile.write("\n")
        outFile.flush()

    
    print("Found: ", found, replyFound)
            
if __name__ == '__main__':
    import os, sys

    args = sys.argv[1:]
    inFile = bz2.open(args[0], 'rb')
    threadIdsFile = args[1]
    replyIdsFile = args[2]

    outFile = args[3]
    
    ids = set([ line.split('"')[1] for line in open(threadIdsFile, "r") ])
    replyIds = set([ line.split('"')[1] for line in open(replyIdsFile, "r") ])

    print("Ids to find: ", len(ids), len(replyIds))
    filterIDs(inFile, ids, replyIds, open(outFile, "w"))