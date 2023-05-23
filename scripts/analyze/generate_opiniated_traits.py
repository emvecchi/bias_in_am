from nltk.corpus import wordnet

if __name__ == '__main__':
    import os, sys

    args = sys.argv[1:]
    outFilename = args[0]

    opiniated = "correct, wrong, stupid,"
    opiniated += "arrogant, bossy, closed-minded, condescending, cynical, defeatist, dishonest, disloyal, dogmatic, flippant, greedy, inconsiderate, inflexible,"
    opiniated += "insincere, intolerant, manipulative, narrow-minded, overbearing, pessimistic, prejudiced, self-centered, selfish, skeptical, snobbish, stubborn,"
    opiniated += "superficial, suspicious, unforgiving, ungrateful, unreliable, unsympathetic, vain, xenophobic,"

    opiniated += "valid, well-reasoned, disagreeable, agreeable, misinformed, baseless, thought-provoking, nonsensical, flawed, question-raising, shortsighted,"
    opiniated += "astute, insightful, questionable, informed, shallow, stimulating, simplistic, ethical, problematic, impressive, naive, persuasive, unconvincing,"
    opiniated += "uninformed, thoughtful, credible, dubious, conclusive, specious, insightful, superficial, well-articulated, preposterous, validating, challenging,"
    opiniated += "inarticulate, sound, hollow, disputable, misguided, foolish, accurate, inaccurate, unreasonable, rational, irrational, logical, illogical,"
    opiniated += "sensible, nonsensible, inconclusive, inconsistent, convincing, unconvincing, well-founded, unsupported, unsubstantiated, misinformed, thoughtless,"
    opiniated += "self-loving, narcissistic, ridiculous, right"

    allPossibs = { }

    for word in opiniated.split(","):
        word = word.strip().lower()

        allPossibs[word] = 1
        for syn in wordnet.synsets(word):
            if syn.pos() in [ "v", "n" ]:
                continue

            for lemma in syn.lemmas():
                allPossibs[lemma.name()]  = 1

                for an in lemma.antonyms():
                    allPossibs[an.name()]  = 1

                    for syn2 in wordnet.synsets(an.name()):
                        if syn2.pos() in [ "v", "n" ]:
                            continue

                        for lemma2 in syn2.lemmas():
                            allPossibs[lemma2.name()]  = 1

    with open(outFilename, "w") as opinionsF:
        for key in sorted(allPossibs.keys()):
            opinionsF.write(key + "\n")