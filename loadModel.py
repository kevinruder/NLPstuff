import spacy
import da_model
import json

nlp = spacy.load("C:/Users/kvru/goldStandardNERDAVec/")

#text = "Der er få navn i denne her tekst, men John Petersen er et navn. Søren Knudsen er også et navn som bliver brugt meget i Danmark. Men det er ofte at tiltalte Ivar Hansen ikke bliver nævnt.  Af og til spiser jeg en banana, men ikke idag. Jon er ikke et navn, det er Jens heller ikke."

    
def extractEntities(text):
    if type(text) == str:
        doc = nlp(text)
        entities =[]
        for ent in doc.ents:
            entities.append(str(ent))
        return json.dumps(entities)

    else:
        return "not a string"

# with open(r"C:\Users\kvru\Downloads\sag2.txt", encoding="utf8") as f:
#     text = f.read().replace('\n', ' ')
#     doc = nlp(text)
#     print(doc)
#     print(doc)


# with open(r"C:\Users\kvru\Downloads\results.txt","w", encoding="utf8") as f:
#     for ent in doc.ents:
#         f.write(str(ent)+'\n')

# spacy.displacy.serve(doc,style='ent')        


