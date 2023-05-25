from lark import Lark, Transformer, Visitor
import string, sys

class GenderExtractor(Transformer):
    
    def GENDER(self, token):
        self.gender = str(token)
        return token
    
    def intro(self, children):
        self.introv = [ str(token) for token in children ]
        return children
    
    def findGender(self, tree):
        self.gender = None
        self.transform(tree)
        return self.gender
    
    def findGenderAndIntro(self, tree):
        self.gender = None
        self.introv = None
        self.transform(tree)
        return self.gender, self.introv
    
class GenderMatcher():
    def __init__(self, dataDir):
        genders = self.__terminals(self.__readTerminals(dataDir + "/genders.txt"))
        features = self.__terminals(self.__readTerminals(dataDir + "/descriptions.txt"))
        expressions = self.__terminals(self.__readTerminals(dataDir + "/expressions.txt"))
        endings = self.__terminals(self.__readTerminals(dataDir + "/endings.txt"))

        self.lark = Lark(f'''start: not_important* identification not_important*

            identification: EXPRESSION description GENDER | BEGINNING description? GENDER  description? ENDING SPACE

            GENDER: {genders}
            FEATURE: {features} | NUMBER | SPACE
            EXPRESSION: {expressions}
            ENDING: {endings}
            BEGINNING: "being" | "as"
            description: FEATURE | FEATURE description
            
            descr: NUMBER | FEATURE
            not_important: /\\w+/

            SPACE: " "
            %import common.WORD   // imports from terminal library
            %import common.NUMBER
            %ignore " "           // Disregard spaces in text
         ''')
        
        self.glark = Lark(f'''start: not_important* intro GENDER not_important*

            intro: EXPRESSION /\\w+/*

            GENDER: {genders}
            not_important: /\\w+/
            SPACE: " "

            EXPRESSION: {expressions}

            %import common.WORD   // imports from terminal library
            %import common.NUMBER
            %ignore " "           // Disregard spaces in text
         ''')

    def findGender(self, text):
        text = self.__clearText(text)
        try:
            tree = self.lark.parse(text)
            extractor = GenderExtractor()
            gender = extractor.findGender(tree)
            return gender
        except:
            return None
        
    def findOnlyGender(self, text):
        text = self.__clearText(text)
        try:
            tree = self.glark.parse(text)
            extractor = GenderExtractor()
            gender, intro = extractor.findGenderAndIntro(tree)
            return gender, intro
        except:
            return None, None

    def __readTerminals(self, filename):
        result = set([ ])
        for line in open(filename):
            line = self.__clearText(line)
            result.add(line)
            result.add("".join(line.split()))

        toSort = [ (len(elem), elem) for elem in result ]
        toSort.sort(reverse=True)
        return [ elem for (_, elem) in toSort ]

    def __clearText(self, text, remove = [ "a", "an", "the" ]):
        text = text.lower()

        result = ""
        valid = string.ascii_letters + string.digits + string.punctuation
        for c in text:
            if c.isalnum() or c in valid or c.isspace():
                result += c

        text = result
        translation_table = str.maketrans(string.punctuation, " " * len(string.punctuation))
        text = text.translate(translation_table)
        words = [ word for word in text.split() if word not in remove ]
        return " ".join(words)

    def __terminals(self, col):
        return "|".join(['"' + elem + '"' for elem in col ])


def runTests(filename, matcher):
    for line in open(filename, "r"):
        line = line.strip()
        if not line:
            continue

        if line[0] == "%":
            continue

        parts = line.split("#")
        text = "#".join(parts[:-1])
        correct = parts[-1]

        correct = correct.strip()
        if correct == "None":
            correct = None
    
        gender = matcher.findGender(text)
        if gender != correct:
            print("Should be: ", correct)
            print("Was:", gender)
            print("Text:", text)
            print()

def runOneTest(matcher, text):
    print(matcher.findGender(text))

def runOneGenderTest(matcher, text):
    print(matcher.findOnlyGender(text))

def parseOnlyGender(matcher, filename):
    words = { }

    for line in open(filename, "r"):
        line = line.strip()

        ogender, intro = matcher.findOnlyGender(line)
        gender = matcher.findGender(line)

        if gender != ogender:
            if ogender is None:
                continue


            print("%", " ".join(intro), ogender)
            print(line, "#", "None", "\n")
            for w in intro:
                if w not in words:
                    words[w] = 1
                else:
                    words[w] += 1
            
       
    words = [ (freq, word) for (word, freq) in words.items() ]
    words.sort(reverse=True)
    print(words[:20])

if __name__ == '__main__':
    import os, sys

    matcher = GenderMatcher("./data/")
    runTests("./data/manual_tests.txt", matcher)
    runTests("./data/tests.txt", matcher)
    
    #runOneTest(matcher, "Please help change my view because I never, never used to feel this way about ANYONE in the LBGT community and now I'm scared positively shitless of bisexual men...")
    #parseOnlyGender(matcher, "./data/sentences_with_genders.txt")

    #runOneGenderTest(matcher, "I am bla bla bla woman")