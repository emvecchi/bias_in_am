import json, bz2
import re

def interesting_title(cmv):
    meta = ["/r/changemyview report",
            "Fresh Topic Friday".lower(),
            "[Mod Cross-Post]".lower(),
            "[Mod ".lower(),
            "[meta]", "[modpost]", "mod post",  "[mod-post]", "tcmv tuesday",
            "sexless saturday",
            "it's fresh topic" ]
    
    title = cmv["title"].lower().strip()

    if any([title.startswith(m) for m in meta ]):
        return False
    
    title = "".join([c for c in title if c not in ".!@#$%^&*()_+=-{}][\":;'?><,./\"']"])

    if not any([title.startswith(b) for b in [ "cmv", "tcmv" ]]) and not any([title.endswith(e) for e in [ "cmv" ]]):
        print("Warning", title)

    return True

if __name__ == '__main__':
    import os, sys

    args = sys.argv[1:]
    trainFile = bz2.open(args[0], 'rb')
    heldoutFile = bz2.open(args[1], 'rb')

    outFile = open(args[2], 'w')

    for file in [ trainFile, heldoutFile]:
        for line in file:
            cmv = json.loads(line)

            if not interesting_title(cmv):
                continue

            json.dump(cmv, outFile)
            outFile.write("\n")
        
        file.close()

    outFile.close()