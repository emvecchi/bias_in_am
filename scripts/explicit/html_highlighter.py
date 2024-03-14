import bz2, json
import projpath

def highlightText(text, phrase, matcher, start = "<mark>", end = "</mark>"):
    result = ""
    clearPhrase = matcher.clearForHighlight(phrase)
    opened = False
    foundToHighlight = False
    for i in range(len(text)):
        letter = text[i]
        if not opened and letter.lower() != "a" and len(matcher.clearForHighlight(letter)) == 0:
            result += text[i]
            continue
               
        if not opened:
            ending = text[i:]
            clearEnding = matcher.clearForHighlight(ending)
            if clearEnding.startswith(clearPhrase):
                result += start
                opened = True
                foundToHighlight = True
        
        if opened:
            beginning = text[:i]
            clearBeginning = matcher.clearForHighlight(beginning)
            if clearBeginning.endswith(clearPhrase):
                result += end
                opened = False

        result += text[i]

    if not foundToHighlight:
        print("Not found!", phrase)

    return result

def generateHtml(data, batchId):
    html = ["<meta charset=\"UTF-8\">",
            "<html>", 
            "<head>",
            "<title>Explicit Gender, batch nr{}</title>".format(batchId),
            "</head>",
            "<body>" ]
    
    genderNice = { "M" : "male", "F" : "female", "N" : "other", "MULT" : "Help needed" }

    for cmv in data:
        html.append("<h1>{}</h1>".format(cmv["id"]))

        if "title" in cmv:
            html.append("<h2>{}</h2>".format(cmv["title"]))

        if cmv["gender"] in genderNice:
            nicer = genderNice[cmv["gender"]]
        else:
            nicer = cmv["gender"]

        html.append("<h3 style=\"color:IndianRed;\">Gender: {}</h3>".format(nicer))
        html.append("<p>{}</p>".format(cmv["text"]))
        html.append("<hr color=\"#2C3E50 \" size=\"10px\" />")

    html += ["</body>", "</html>" ]
    return "\n".join(html)
