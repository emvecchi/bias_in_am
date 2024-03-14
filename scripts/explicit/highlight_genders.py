import bz2, json
import projpath
from explicit.gender_matcher import GenderMatcher
import markdown
from common.normalize import normalizeTextForParse
import html_highlighter

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

def selectBestGender(text, matcher):
    genders = set([ ])
    for line in text.split("\n"):
        gender = matcher.findGender(line)

        if gender is None:
            continue

        gID = THIRD_GENDER
        if gender in GENDERS:
            gID = GENDERS[gender]
            
        genders.add(gID)

    if len(genders) != 1:
        return UNKNOWN
    else:
        return list(genders)[0]

def highlightText(text, matcher):
    result = [ ]
    foundGenders = set([ ])
    for line in text.split("\n"):
        clean_text = normalizeTextForParse(line)
        
        if not clean_text:
            result.append(line)
            continue
        
        gender = matcher.findGender(clean_text)

        if gender is None:
            result.append(line)
            continue

        gID = THIRD_GENDER
        if gender in GENDERS:
            gID = GENDERS[gender]
            
        foundGenders.add(gID)
        
        identification = matcher.findIdentification(line)
        if identification is None:
            print("Something went wrong, no identification!")
            print("Line: ", line)
            print("Gender: ", gender)
            result.append(line)
            continue

        highlighted = html_highlighter.highlightText(line, identification, matcher)
        result.append(highlighted)
    
    # more than one gender or none
    if len(foundGenders) != 1:
        return None
        
    mainGender = list(foundGenders)[0]
    finalText = markdown.markdown("\n".join(result))
    return finalText, mainGender


def highlightThreads(inFile, matcher):
    outData = [ ]
    for entry in inFile:
        cmv = json.loads(entry)
        highlighted = highlightText(cmv["selftext"], matcher)
        if highlighted is None:
            continue

        info = { "id" : cmv["id"],
                 "title" : cmv["title"],
                 "text" : highlighted[0],
                 "gender" : highlighted[1]}

        outData.append(info)

    return outData

def highlightResponses(inFile, matcher):
    outData = [ ]
    for entry in inFile:
        cmv = json.loads(entry)

        for comment in cmv["comments"]:
            if "body" not in comment:
                continue

            highlighted = highlightText(comment["body"], matcher)
            if highlighted is None:
                continue

            info = { "id" : cmv["id"],
                 "text" : highlighted[0],
                 "gender" : highlighted[1]}

            outData.append(info)
            
    return outData

if __name__ == '__main__':
    import os, sys

    args = sys.argv[1:]
    inFile = open(args[0], 'r')
    what = args[1]
    outDir = args[2]

    thisPath = os.path.dirname(os.path.realpath(__file__))
    matcher = GenderMatcher(thisPath + "/data/")

    if what == "threads":
        outData = highlightThreads(inFile, matcher)
    elif what == "responses":
        outData = highlightResponses(inFile, matcher)
    else:
        print("Unknown type:", what)
        sys.exit()

    batch_size = 50
    batches = [outData[i:i+batch_size] for i in range(0, len(outData), batch_size)]

    # Print the batches
    for bId, batch in enumerate(batches):
        outFile = outDir + "/" + "explicit_gender." + what + ".batch_" + str(bId+1) + ".html"
        html = html_highlighter.generateHtml(batch, bId+1)
        out = open(outFile, "w")
        out.write(html)
        out.close()

    
    