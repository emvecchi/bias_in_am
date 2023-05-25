import bz2, json
import projpath
from match_explicit_gender import GenderMatcher
from preprocess import filter_json_fields

GENDERS = { "man" : "M",
                "men" : "M",
                "woman" : "F",
                "women" : "F",
                "girl" : "F",
                "boy" : "M",
                "female" : "F",
                "male" : "M",
                "guy" : "M",
                "cis man" : "M",
                "cis woman" : "F",
                "cisgendered male" : "M",
                "cisgender male" : "M",
                "cisgendered female" : "F",
                "cisgender female" : "F",
                "cis man" : "M",
                "cis woman" : "F",
                "cis female" : "F",
                "cis male" : "M",
                "cisgendered man" : "M",
                "cisgender man" : "M",
                "cisgendered woman" : "F",
                "cisgender woman" : "F" }
    
UNKNOWN = "UNK"
THIRD_GENDER = "N"

def selectBestGender(cmvId, text, matcher):
    genders = set([ ])
    for line in text.split("\n"):
        gender = matcher.findGender(line)

        if gender is None:
            continue

        gID = THIRD_GENDER
        if gender in GENDERS:
            gID = GENDERS[gender]
            
        genders.add(gID)

    if len(genders) == 0:
        return UNKNOWN
    elif len(genders) > 1:
        print("More than one gender", genders, cmvId)
        return UNKNOWN
    else:
        return list(genders)[0]

def annotateGenders(inFile, matcher, outThreads, outReplies, outAuthors):
    for entry in inFile:
        cmv = json.loads(entry)
        cmvGender = selectBestGender(cmv["id"], cmv["selftext"], matcher)

        outThreads.write(cmv["id"] + '\t' + cmvGender + "\n")

        if cmvGender != UNKNOWN:
            outAuthors.write(cmv["author"] + '\t' + cmvGender + "\n")

        for comment in cmv["comments"]:
            if "body" not in comment:
                comGender = UNKNOWN
            else:
                comGender = selectBestGender(comment["id"], comment["body"], matcher)

            outReplies.write(comment["id"] + '\t' + comGender + "\n")

            if "author" in comment and comGender != UNKNOWN:
                outAuthors.write(comment["author"] + '\t' + comGender + "\n")
            
if __name__ == '__main__':
    import os, sys

    args = sys.argv[1:]
    inFile = bz2.open(args[0], 'rb')

    outThreads = open(args[1], 'w')
    outReplies = open(args[2], 'w')
    outAuthors = open(args[3], 'w')

    thisPath = os.path.dirname(os.path.realpath(__file__))
    matcher = GenderMatcher(thisPath + "/data/")
    annotateGenders(inFile, matcher, outThreads, outReplies, outAuthors)

    outThreads.close()
    outReplies.close()
    outAuthors.close()