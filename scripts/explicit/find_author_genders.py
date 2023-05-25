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
                "cis woman" : "F" }
    
UNKNOWN = "UNK"
THIRD_GENDER = "N"

def filterGenderSentences(inFile, out):
    matcher = GenderMatcher("./data/")
    for entry in inFile:
        cmv = json.loads(entry)
    
        genders = set([ ])
        for line in cmv["selftext"].split("\n"):
            gender = matcher.findGender(line)

            if gender is None:
                continue

            if gender in GENDERS:
                gID = GENDERS[gender]
            else:
                gID = THIRD_GENDER

            genders.add(gID)

        if len(genders) == 0:
            continue

        if len(genders) > 1:
            print("More than one gender", genders, cmv["id"])
            continue
        else:
            cmvGender = list(genders)[0]

        #cmv = filter_json_fields.cleanCMV(cmv)
        cmv["gender"] = cmvGender
        json.dump(cmv, out)
        out.write("\n")
        out.flush()
        
if __name__ == '__main__':
    import os, sys

    args = sys.argv[1:]
    inFile = bz2.open(args[0], 'rb')

    out = open(args[1], 'w')
    filterGenderSentences(inFile, out)