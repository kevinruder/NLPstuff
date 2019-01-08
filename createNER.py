import spacy

nlp = spacy.load('C:/Users/kvru/DA_vectors_lg')
nlp.add_pipe(nlp.create_pipe('ner'))
nlp.begin_training()
nlp.to_disk(r'C:\Users\kvru\DAvectorsWITHEMPTYNER')