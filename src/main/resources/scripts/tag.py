import spacy, tqdm

import fire

TAGS = ['noun', 'pronoun', 'verb', 'adjective', 'adverb', 'det', 'numeral', 'other']
TMAP = {
    # coarse grained, https://universaldependencies.org/u/pos/

    'ADJ': 'adjective',
    'ADP': 'other',
    'ADV': 'adverb',
    'AUX': 'verb',
    'CCONJ': 'other',
    'DET': 'det',
    'INTJ': 'other',
    'NOUN': 'noun',
    'NUM': 'numeral',
    'PART': 'other',
    'PRON': 'pronoun',
    'PROPN': 'noun',
    'PUNCT': 'skip',
    'SCONJ': 'other',
    'SYM': 'skip',
    'VERB': 'verb',
    'X': 'other',

    # fine grained
    '$': 'skip',
    "''" : 'skip',
    '-LRB -': 'skip',
    '-RRB -': 'skip',
    '.':'skip',
    ':':'skip',
    'ADD':'skip',
    'AFX':'other',
    'CC':'other',
    'CD':'numeral',
    'DT':'det',
    'EX':'det',
    'FW':'other',
    'HYPH':'skip',
    'IN':'other',
    'JJ':'adjective',
    'JJR':'adjective',
    'JJS':'adjective',
    'LS':'skip',
    'MD':'verb',
    'NFP':'noun',
    'NN':'noun',
    'NNP':'noun',
    'NNPS':'noun',
    'NNS':'noun',
    'PDT':'other',
    'POS':'other',
    'PRP':'noun',
    'PRP$':'noun',
    'RB':'adverb',
    'RBR':'adverb',
    'RBS':'adverb',
    'RP':'adverb',
    'SYM':'skip',
    'TO':'other',
    'UH':'other',
    'VB':'verb',
    'VBD':'verb',
    'VBG':'verb',
    'VBN':'verb',
    'VBP':'verb',
    'VBZ':'verb',
    'WDT':'other',
    'WP':'other',
    'WP$':'other',
    'WRB':'other',
    'XX':'other',
    '_SP':'skip',
    '``':'skip',


    'SPACE' : 'skip',
}

def tag_book(inputfile, outfile='result.txt', limit=None, language='en', debug=False):

    with open(inputfile, 'r') as file:
        text = file.read()

    if language == 'en':
        nlp = spacy.load('en_core_web_sm')
    elif language == 'de':
        nlp = spacy.load('de_core_news_sm')
    elif language == 'it':
        nlp = spacy.load('it_core_news_sm')
    elif language == 'es':
        nlp = spacy.load('es_core_news_sm')
    elif language == 'nl':
        nlp = spacy.load('nl_core_news_sm')
    else:
        raise

    nlp.max_length = 2_000_000
    doc = nlp(text)

    calls = 0
    buffer = ''
    result = ''
    for token in doc:
        word, tag = token.text, token.pos_ # nb: we're using coarse grained tags
        assert tag in TMAP, f'{tag}: {spacy.explain(tag)}'

        tag = TMAP[tag]
        if tag == 'skip':
            continue

        word = word.lower().replace('.', '')
        if len(word) == 0:
            continue

        result += ' ' + word.lower().replace('.', '') + '.' + tag

    with open(outfile, 'w') as out:
        out.write(result)

    print('done.')

def test_result(inputfile='result.txt'):
    with open(inputfile, 'r') as file:
        text = file.read()

    tokens = text.split()

    for token in tokens:
        assert len(token.split('.')) == 2, f'{token}'

        word, tag = token.split('.')

        assert tag in TAGS, f'{tag}'
        assert len(word) > 0, f'{word}'
        assert len(word) < 40, f'{word}'

def explain():
    for tag in TMAP.keys():
        print(tag, spacy.explain(tag))

if __name__ == '__main__':
    fire.Fire()