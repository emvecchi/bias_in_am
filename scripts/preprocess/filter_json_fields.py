import json

if __name__ == '__main__':
    import os, sys

    args = sys.argv[1:]
    inFile = open(args[0], 'r')

    outThreads = open(args[1], "w")
    outReplies = open(args[2], "w")

    cmvKeys = ['id', 'author', "author_flair_text",
                   'title', 
                   'selftext',
                   'likes',  
                   'clicked', 
                   'score', 
                   'downs', 
                   'ups', 
                   'num_comments', 
                   'visited' ]

    commentKeys = ['id', 'author', 'name', "author_flair_text",
                    'body', 
                    'likes',
                    'gilded', 
                    'parent_id', 
                    'score', 
                    'controversiality', 
                    'downs', 
                    'ups']

    for line in inFile:
        cmv = json.loads(line)

        newCmv = { }
        for key in cmvKeys:
            newCmv[key] = cmv[key]

        json.dump(newCmv, outThreads)
        outThreads.write("\n")

        for comment in cmv["comments"]:
            newComment = { }
            for key in commentKeys:
                if key in comment:
                    newComment[key] = comment[key]

            json.dump(newComment, outReplies)
            outReplies.write("\n")
       
    outThreads.close()  
    outReplies.close()