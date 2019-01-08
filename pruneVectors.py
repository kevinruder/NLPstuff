import spacy

nlp = spacy.load(r'C:\Users\kvru\DAvectorsWITHEMPTYNER')
removedVectors = nlp.vocab.prune_vectors(20000)
nlp.begin_training()
print(len(nlp.vocab.vectors))
nlp.to_disk(r'C:\Users\kvru\DAvectorsWITHEMPTYNER')
print("Hi")