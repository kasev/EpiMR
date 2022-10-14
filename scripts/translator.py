import json
import requests

publicfolder = "8fe7d59de1eafe5f8eaebc0044534606"
morpheus_by_lemma = json.loads(requests.get("https://sciencedata.dk/public/" + publicfolder + "/morpheus_by_lemma.json").content)


def translator(word):
    try:
        translation = morpheus_by_lemma[word][0]["s"]
    except:
        translation = ""
    return translation