# coding: utf8
from __future__ import unicode_literals

from pathlib import Path
from spacy.util import load_model_from_init_py, get_model_meta
from spacy.language import Language
from spacy.tokens import Span
from spacy.matcher import Matcher
import ujson



__version__ = get_model_meta(Path(__file__).parent)['version']


def load(**overrides):
    Language.factories['entity_matcher'] = lambda nlp, **cfg: EntityMatcher(nlp, **cfg)
    return load_model_from_init_py(__file__, **overrides)


class EntityMatcher(object):
    name = 'entity_matcher'

    def __init__(self, nlp, **cfg):
        Span.set_extension('via_patterns', default=False)
        self.filename = 'patterns.json'
        self.patterns = {}
        self.matcher = Matcher(nlp.vocab)

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []
        for match_id, start, end in matches:
            span = Span(doc, start, end, label=match_id)
            span._.via_patterns = True
            spans.append(span)
        doc.ents = list(doc.ents) + spans
        return doc

    def from_disk(self, path, **cfg):
        patterns_path = path / self.filename
        with patterns_path.open('r', encoding='utf8') as f:
            self.from_bytes(f)
        return self

    def to_disk(self, path):
        patterns = self.to_bytes()
        patterns_path = Path(path) / self.filename
        patterns_path.open('w', encoding='utf8').write(patterns)

    def from_bytes(self, bytes_data):
        self.patterns = ujson.load(bytes_data)
        for label, patterns in self.patterns.items():
            self.matcher.add(label, None, *patterns)
        return self

    def to_bytes(self, **cfg):
        return ujson.dumps(self.patterns, indent=2, ensure_ascii=False)