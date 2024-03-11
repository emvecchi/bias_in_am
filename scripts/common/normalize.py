from bs4 import BeautifulSoup
import re

def normalizeTextForParse(text):
    text = text.strip()
    if not text:
        return None
    
     # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()

    # Remove URLs using regular expressions
    clean_text = re.sub(r"\(http\S+\)|\(www\S+\)|\(https\S+\)", " [LINK] ", clean_text)

    # Remove non-alphanumeric characters and convert to lowercase
    #clean_text = re.sub(r"[^a-zA-Z0-9]", " ", clean_text.lower())

    if clean_text.strip().startswith("&gt;") or clean_text.strip().startswith(">"):
        return None
    
    return clean_text