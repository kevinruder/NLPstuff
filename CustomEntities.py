#!/usr/bin/env python

# coding: utf8

"""Example of a spaCy v2.0 pipeline component that sets entity annotations

based on list of single or multiple-word company names. Companies are

labelled as ORG and their spans are merged into one token. Additionally,

._.has_tech_org and ._.is_tech_org is set on the Doc/Span and Token

respectively.



* Custom pipeline components: https://spacy.io//usage/processing-pipelines#custom-components



Compatible with: spaCy v2.0.0+

"""

from __future__ import unicode_literals, print_function


import csv
import plac
import glob

import spacy

from spacy.tokenizer import Tokenizer

from spacy.language import Language

import lemmy.pipe

from spacy.matcher import PhraseMatcher

from spacy.tokens import Doc, Span, Token

from spacy.lang.da.stop_words import STOP_WORDS



@plac.annotations(

    text=("Text to process", "positional", None, str),

    danish_names=("Names of technology companies", "positional", None, str))

def main(text="Jens er ikke så klog, men han er kloger end Kristoffer", *danish_names):

    # For simplicity, we start off with only the blank English Language class

    # and no model or pre-defined pipeline loaded.

    nlp = spacy.load('C:/Users/kvru/DA_vectors_sm')

    if not danish_names:  # set default companies if none are set via args
        
        names_list = []
        with open('C:/Users/kvru/Downloads/drengenavn2.csv') as f:
            reader = csv.reader(f,delimiter=',')
            for row in reader:
                for name in row:
                    names_list.append(name)

        
    for stopword in STOP_WORDS:
        nlp.vocab[stopword].is_stop = True    
            

    component = TechCompanyRecognizer(nlp, names_list)  # initialise component

    Language.factories['danish_names'] = lambda nlp, **cfg: TechCompanyRecognizer(nlp, **cfg)

    # lemmatizer = lemmy.pipe.load()

    # Language.factories['lemmy'] = lambda nlp, **cfg: lemmatizer(nlp.vocab, **cfg)

    # add the comonent to the spaCy pipeline.
    # nlp.add_pipe(lemmatizer, last=True)

    # lemmas can now be accessed using the `._.lemma` attribute on the tokens
    nlp.add_pipe(component, last=True)  # add last to the pipeline

    TRAIN_DATA = []

    file_directory = glob.glob("C:/Users/kvru/Downloads/Nytår/*.txt")

    for txt in file_directory:
        with open(txt) as f:
            content = f.readlines()
            for line in content:
                #Remove stop words
                temp_doc = nlp(line)
                tokens = [token.text for token in temp_doc]
                tokens = [token for token in tokens if nlp.vocab[token].is_stop == False]
                tokens = ' '.join(tokens)
                doc = nlp(tokens)
                temp_train_data = [(e.doc.text,{'entities':[(e.start_char,e.end_char,e.label_)]}) for e in doc.ents]
                for t in temp_train_data:
                    TRAIN_DATA.append(t)


    
    #doc = nlp(text)

    # print('Pipeline', nlp.pipe_names)  # pipeline contains component name

    # print('Tokens', [t.text for t in doc])  # company names from the list are merged

    # print('Doc has_danish_names', doc._.has_danish_name)  # Doc contains tech orgs

    # print('Token 0 is_danish_name', doc[0]._.is_danish_name)  # "Alphabet Inc." is a tech org

    # print('Token 1 is_danish_name', doc[1]._.is_danish_name)  # "is" is not

    # print('Entities', [(e.text, e.label_) for e in doc.ents])  # all orgs are entities

    # print('Entities', [(e.text, e.label_) for e in doc.ents])  # all orgs are entities

    
    #TRAIN_DATA = [(e.doc,{'entities':[(e.start_char,e.end_char,e.label_)]}) for e in doc.ents]

    import random

    from pathlib import Path

    from spacy.util import minibatch, compounding

    if 'ner' not in nlp.pipe_names:

        ner = nlp.create_pipe('ner')

        nlp.add_pipe(ner, last=True)

    # otherwise, get it so we can add labels

    else:

        ner = nlp.get_pipe('ner')

    n_iter = 20
    output_dir = 'C:/Users/kvru/DA_model'


    # add labels    

    for _, annotations in TRAIN_DATA:

        for ent in annotations.get('entities'):
        
            ner.add_label(ent[2])



    # get names of other pipes to disable them during training

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    with nlp.disable_pipes(*other_pipes):  # only train NER

        optimizer = nlp.begin_training()

        for itn in range(n_iter):

            random.shuffle(TRAIN_DATA)

            losses = {}

            # batch up the examples using spaCy's minibatch

            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))

            for batch in batches:

                texts, annotations = zip(*batch)

                nlp.update(

                    texts,  # batch of texts

                    annotations,  # batch of annotations

                    drop=0.5,  # dropout - make it harder to memorise data

                    sgd=optimizer,  # callable to update weights

                    losses=losses)

            print('Losses', losses)



    # test the trained model

    for text, _ in TRAIN_DATA:

        doc = nlp(text)

        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])

        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])



    # save model to output directory

    if output_dir is not None:

        output_dir = Path(output_dir)

        if not output_dir.exists():

            output_dir.mkdir()

        nlp.to_disk(output_dir)

        print("Saved model to", output_dir)



        # test the saved model

        print("Loading from", output_dir)

        nlp2 = spacy.load(output_dir)

        for text, _ in TRAIN_DATA:

            doc = nlp2(text)

            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])

            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])    
    # res= []
    # to_train_ents = []
    # with open('gina_haspel.txt') as gh:
    # line = True
    # while line:
    #     line = gh.readline()
    #     mnlp_line = nlp(line)
    #     matches = matcher(mnlp_line)
    #     res = [Offsetter(Label, mnlp_line,x) for x in matches]
    #     to_train_ents.append((line,dict(entities=res)))





class TechCompanyRecognizer(object):

    """Example of a spaCy v2.0 pipeline component that sets entity annotations

    based on list of single or multiple-word company names. Companies are

    labelled as ORG and their spans are merged into one token. Additionally,

    ._.has_tech_org and ._.is_tech_org is set on the Doc/Span and Token

    respectively."""

    name = 'danish_names'  # component name, will show up in the pipeline

    def Offsetter(self,lbl,doc,matchitem):
        o_one = len(str(doc[0:matchitem[1]]))
        subdoc = doc[matchitem[1]:matchitem[2]]
        o_two = o_one + len(str(subdoc))
        return (o_one,o_two,lbl)
    
    

    def __init__(self, nlp, companies=tuple(), label='PERSON'):

        """Initialise the pipeline component. The shared nlp instance is used

        to initialise the matcher with the shared vocab, get the label ID and

        generate Doc objects as phrase match patterns.

        """

        self.label = nlp.vocab.strings[label]  # get entity label ID



        # Set up the PhraseMatcher – it can now take Doc objects as patterns,

        # so even if the list of companies is long, it's very efficient

        patterns = [nlp(org) for org in companies]

        self.matcher = PhraseMatcher(nlp.vocab)

        self.matcher.add('DANISH_NAMES', None, *patterns)



        # Register attribute on the Token. We'll be overwriting this based on

        # the matches, so we're only setting a default value, not a getter.

        Token.set_extension('is_danish_name', default=False)



        # Register attributes on Doc and Span via a getter that checks if one of

        # the contained tokens is set to is_tech_org == True.

        Doc.set_extension('has_danish_name', getter=self.has_tech_org)

        Span.set_extension('has_danish_name', getter=self.has_tech_org)



    def __call__(self, doc):

        """Apply the pipeline component on a Doc object and modify it if matches

        are found. Return the Doc, so it can be processed by the next component

        in the pipeline, if available.

        """

        matches = self.matcher(doc)

        spans = []  # keep the spans for later so we can merge them afterwards

        for _, start, end in matches:

            # Generate Span representing the entity & set label

            entity = Span(doc, start, end, label=self.label)

            spans.append(entity)

            # Set custom attribute on each token of the entity

            for token in entity:

                token._.set('is_danish_name', True)

            # Overwrite doc.ents and add entity – be careful not to replace!

            doc.ents = list(doc.ents) + [entity]

        for span in spans:

            # Iterate over all spans and merge them into one token. This is done

            # after setting the entities – otherwise, it would cause mismatched

            # indices!

            span.merge()

        return doc  # don't forget to return the Doc!



    def has_tech_org(self, tokens):

        """Getter for Doc and Span attributes. Returns True if one of the tokens

        is a tech org. Since the getter is only called when we access the

        attribute, we can refer to the Token's 'is_tech_org' attribute here,

        which is already set in the processing step."""

        return any([t._.get('is_danish_name') for t in tokens])





if __name__ == '__main__':

    plac.call(main)



    # Expected output:

    # Pipeline ['danish_names']

    # Tokens ['Alphabet Inc.', 'is', 'the', 'company', 'behind', 'Google', '.']

    # Doc has_tech_org True

    # Token 0 is_tech_org True

    # Token 1 is_tech_org False

    # Entities [('Alphabet Inc.', 'ORG'), ('Google', 'ORG')]