import spacy
import 

nlp = spacy.load(r'C:\Users\kvru\blankDAModel')
nlp.add_pipe(nlp.create_pipe('ner'))
nlp.begin_training()
nlp.to_disk(r'C:\Users\kvru\blankDAModelWITHNER')