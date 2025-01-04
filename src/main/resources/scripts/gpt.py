from openai import OpenAI
import spacy, tqdm

import fire

TAGS = ['noun', 'pronoun', 'verb', 'adjective', 'adverb', 'article', 'numeral', 'other']

def call(text, debug=False):

    if debug:
        return '<call>' + text +  '</call>'

    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content":
                    f"""Please pre-process and POS-tag the following text, using the set of tags: noun, pronoun, verb, adjective, adverb, article, numeral and other. The text should be lowercased and stripped of punctuation. When in doubt, pick the most applicable tag. Provide the result as unformatted text with the format "the.adjective cat.noun walked.verb". 
    Make sure there is no whitespace between the word and the POS tag, only a period. Contractions like "we'll" and "I've" and "didn't" should be broken up into their components ("we" and "ll", "i" and "ve", "did" and "nt") and each component should be tagged separately. For example, "He didn't say, we'd walk to the store." should be tagged as "he.pronoun did.verb nt.other say.verb we.pronoun d.verb walk.verb to.other the.article store.noun". 
    Please check the following carefully.
     * When split by whitespace, each token should contain exactly one period (.) with only alphanumeric characters before and after the period.
     * After the period, there should only ever be a tag from the set of allowed tags. No other tags are allowed.
     * Before the period there should be a sequence of alphanumeric,lowercase characters, corresponding as closely as possible to the original word, or subword, in the text.  

    "{text}" """
            }
        ]
    )

    return completion.choices[0].message.content

def tag_book(inputfile, outfile='result.txt', limit=None, language='en', charpercall=5000, debug=False):

    with open(inputfile, 'r') as file:
        text = file.read()

    if language == 'en':
        nlp = spacy.load('en_core_web_sm')
    elif language == 'de':
        nlp = spacy.load('de_core_news_sm')
        nlp.max_length = 2_000_000
    else:
        raise

    doc = nlp(text)
    sentences = [sentence.text for sentence in tqdm.tqdm(doc.sents)]

    print('Sentence tokenization finished.')

    calls = 0
    buffer = ''
    result = ''
    for sentence in tqdm.tqdm(sentences):

        if len(buffer) + len(sentence) > charpercall:
            # make the API call
            result += ' ' + call(buffer, debug)
            buffer = ''
            calls += 1

            if limit is not None and calls >= limit:
                break

        buffer += ' ' + sentence

    if len(buffer.strip()) > 0:
        result += ' ' + call(buffer, debug)

    with open(outfile, 'w') as out:
        out.write(result)

    print('done.')

def test_result(inputfile):
    with open(inputfile, 'r') as file:
        text = file.read()

    tokens = text.split()

    for token in tokens:
        assert len(token.split('.')) == 2, f'{token}'

        word, tag = token.split('.')

        assert tag in TAGS, f'{tag}'
        assert len(word) > 0, f'{word}'
        assert len(word) < 24, f'{word}'


if __name__ == '__main__':
    fire.Fire()