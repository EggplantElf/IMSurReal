from collections import defaultdict
from itertools import repeat, cycle
import random
import re
import lzma
import Levenshtein

class Token(dict):
    def __init__(self, entries):
        self.vecs = {}
        for k, v in entries.items():
            self[k] = v

    def __hash__(self):
        return self['tid']

    def __repr__(self):
        # return f"<{self['tid']}, {self['olemma']}, {self['hid']}>"
        # return f"{self['tid']}({self['original_id']})"
        return f"{(self['original_id'], self['olemma']) }"

    def __lt__(self, a):
        return self['original_id'] < a['original_id']

    def not_empty(self):
        return self['lemma'] != '_' or self['upos'] != 'PRON'

    def convert_lemma_morph(self):
        # convert korean 
        self['clemma'] = self['lemma'] # used for the characters in inflection
        stem = self['lemma'].split('+')[0]
        if stem:
            self['lemma'] = stem # use as feature
        xmorph = self['xpos'].split('+')
        self['morph'] += xmorph

    def get_diff(self):
        self['diff'] = get_edit_diff(self['clemma'], self['word'])

class Root(Token):
    def __init__(self, entries):
        super().__init__(entries)


class LostToken:
    # for generating new tokens
    def __init__(self, signature):
        self.lemma, self.upos, self.morphstr = signature
        self.morph = [] if self.morphstr == '_' else self.morphstr.split('|')

    def generate(self, hid=None):
        t =  Token({'tid': None,
                    'oword': self.lemma,
                    'word': self.lemma,
                    'cword': '_',
                    'olemma': self.lemma,
                    'lemma': self.lemma,
                    'clemma': self.lemma,
                    'upos': self.upos,
                    'xpos': '_',
                    'hid': hid,
                    'original_id': None,
                    'morph': self.morph,
                    'omorphstr': self.morphstr,
                    'label': '<LOST>',
                    'oids': []
                    })
        return t

class Sentence(dict):
    def __init__(self):
        self.meta = ''
        self['gold_contracted_tokens'] = [] # contracted token from ud file
        self['gold_contracted_lines'] = {} # contracted token from ud file
        self.lost = [] # gold lost token in T2
        self.tokens = [Root({'tid': 0,
                            'word': '<root>',
                            'oword': '<ROOT>',
                            'olemma': '<ROOT>',
                            'lemma': '<root>',
                            'clemma': '<root>',
                            'cword': '', # word form after contraction
                            'upos': '<ROOT>',
                            'hid': -1,
                            'head': None,
                            'phead': None,
                            'phid': None,
                            'original_id': 0,
                            'deps': [],
                            'lost': [],
                            'gold_linearized_domain': [],
                            'linearized_domain': [],
                            'domain': []})]
        self.root = self.tokens[0]
        self.is_projective = True
        self['nonproj_arcs'] = []
        self.root['domain'].append(self.root) # root is in its own domain


        self['input_tokens'] = []      # input, assume unordered
        self['linearized_tokens'] = [] # lin/tsp output
        self['nbest_linearized_tokens'] = [] # nbest lin/tsp output
        self['sorted_tokens'] = []     # swap output
        self['generated_tokens'] = []  # gen output
        self['inflected_tokens'] = []  # inf output
        self['contracted_tokens'] = [] # add contracted token (e.g. 'zu', 'dem' -> 'zum') in the list,
                                            # the tid of the contracted token is '2-3' if tid of 'zu' is 2 and tid of 'dem' is 3
                                            # and the compoments 'zu' and 'dem' are marked with 'contracted' = True,
                                            # so that they won't appear in the output text, but still appear in the lines
        self['gold_linearized_tokens'] = []
        self['gold_generated_tokens'] = []

    def __repr__(self):
        return ' '.join([f"{t['lemma']}({t['tid']})" for t in self.tokens[1:]])

    def clear_pred(self):
        for key in ['linearized', 'sorted', 'generated', 'inflected', 'contracted', 'nbest_linearized']:
            self[f'{key}_tokens'] = []
        for token in self.tokens:
            # del token.vecs
            token.vecs = {}
            token['linearized_domain'] = []
            token['generated_domain'] = []


    def get_tokens(self, include_empty = True):
        return [t for t in self.tokens[1:] if include_empty or t.not_empty()]

    def add_token(self, token):
        self.tokens.append(token)

    def complete(self):
        # whatever needed to do after reading the sentence` 
        # e.g. check for projectivity

        self['input_tokens'] = self.get_tokens(False)
        for t in self.tokens[1:]:
            # encode empty tokens
            h = self.tokens[t['hid']] 
            t['head'] = h
            h['deps'].append(t)
            # not decode empty tokens
            if t.not_empty():
                t['domain'].append(t)
                h['domain'].append(t)

        for t in self.lost:
            t['head'] = self.tokens[t['hid']] 
            t['head']['lost'].append(t)


        # second pass to sort the domains if the original id is known (for training T1)
        if any(t['original_id'] is not None for t in self.tokens[1:]):
            self['gold_linearized_tokens'] = sorted([t for t in self.tokens[1:] if t.not_empty()], \
                                                    key=lambda x: x['original_id']) 
            self['gold_generated_tokens'] = sorted([t for t in self.tokens[1:]+self.lost if t.not_empty()], \
                                                    key=lambda x: x['original_id']) 
            for t in self.tokens:
                t['gold_linearized_domain'] = sorted(t['domain'], key=lambda x:x['original_id'])
                t['gold_generated_domain'] = sorted(t['domain']+t['lost'], key=lambda x:x['original_id'])
            # projective order for swap sort
            self['gold_projective_tokens'] = flatten(self.root, 'gold_linearized_domain')


        # TODO: check projectivity only for swap
        # if self['gold_linearized_tokens']:
        #     # TODO: change for deep input 
        #     try:
        #         self.check_proj()
        #     except:
                # pass

        # gather contracted tokens (only appear in UD file, thus gold tid)
        for line in self['gold_contracted_lines'].values():
            items = line.split('\t')
            word = items[1].lower()
            ids = items[0].split('-')
            b, e = int(ids[0]), int(ids[1])
            tokens = [self.tokens[i] for i in range(b, e+1)]
            # self['gold_contracted_tokens'].append((word, tokens))
            tokens[0]['cword'] = word
            for t in tokens[1:]:
                t['cword'] = ''

        self['gold_contracted_tokens'] = [t for t in self['gold_linearized_tokens'] if t['cword'] not in [' ', '_']]


        # linear order constraints in the input
        for t in self.tokens:
            t['l_order'], t['r_order'] = [], []
            for dep in t['deps']:
                if dep['lin']: 
                    if dep['lin'] < 0:
                        t['l_order'].append(dep)
                    else:
                        t['r_order'].append(dep)
            t['l_order'].sort(key=lambda x:x['lin'])
            t['r_order'].sort(key=lambda x:x['lin'])

            if t['l_order'] + t['r_order']:
                t['order'] = t['l_order'] + [t] + t['r_order']
            else:
                t['order'] = []

    def get_output_tokens(self):
        return  self['contracted_tokens'] or \
                self['generated_tokens'] or \
                self['sorted_tokens'] or \
                self['linearized_tokens'] or \
                self['input_tokens'] or \
                self.tokens[1:]

    def is_proj_arc(self, h, d):
        b, e = min(h, d, key=lambda x:x['original_id']), max(h, d, key=lambda x:x['original_id'])
        for j in range(b['original_id'] + 1, e['original_id']):
            tokens = [self.root] + self['gold_linearized_tokens']
            t = tokens[j]
            while t is not self.root:
                t = t['head']
                if t['original_id'] < b['original_id'] or t['original_id'] > e['original_id']:
                    return False
                elif t is b or t is e:
                    break
        return True

    def check_proj(self):
        for d in self.tokens[1:]:
            h = d['head']
            if not self.is_proj_arc(h, d):
                self['nonproj_arcs'].append((h, d))
                self.is_projective = False
        return self.is_projective

    def get_size(self):
        return total_size(self, verbose=False)


def normalize(word):
    # special repr for words with a number in it, following Schmaltz
    if re.findall('\d', word):
        return '<NUM>'
    else:
        return word


# default read non-ud format as input, which means lemma and word are swapped 
def read_conllu(filename, ud=False, skip_lost=True, orig_word=False, convert_lemma=False, simple=False, endless=False, first=None):
    count = 0
    sent = Sentence()
    with (lzma.open(filename, "rt", encoding='utf-8') if filename.endswith('xz') else open(filename)) as fin:
        while True:
            line = fin.readline()
            if line == '': # end of file
                # endless mode
                if endless:
                    fin.seek(0)
                else:
                    break
            elif line.strip():
                if line.startswith('#'):
                    sent.meta += line
                    if line.startswith('# text = '):
                        sent.text = line.strip().split('# text = ')[1]
                else:
                    entries = line.strip().split('\t')
                    if '-' in entries[0]:
                        sent['gold_contracted_lines'][int(entries[0].split('-')[0])] = line
                    elif '.' in entries[0]:
                        pass
                    else:
                        # separate morphological information from task related information
                        lin = None          # used in both T1 and T2 to indicate the relative position to its head, 
                                            # appear in train and dev, positive means after the head
                        original_id = int(entries[0])
                        ids = []            # used in T2 to indicate all corresponding tokens (head and deleted children) in T1,
                                            # only in train set

                        oword = entries[1] if (ud or entries[2] == '_') else entries[2]
                        olemma = entries[2] if (ud and entries[2] != '_') else entries[1]
                        # might be a bug in the alignment cause empty cword instead of '_'
                        cword = oword.lower() if (ud or entries[8] == '_' or entries[8] == ' ') else entries[8] 
                        morphstr = entries[5]
                        morph = []
                        if morphstr != '_':
                            for items in morphstr.split('|'):
                                k, v = items.split('=')
                                if k == 'original_id':
                                    original_id = int(v)
                                elif k == 'lin':
                                    lin = int(v)
                                elif k.startswith('id'):
                                    ids.append(int(v))
                                else:
                                    # real morphological information
                                    # morph_dict[k] = v
                                    morph.append(items) # key-value pair as a string

                        # VERY IMPORTANT !!! (A BUG IN THE DATA)
                        morph.sort()

                        token = Token({'tid': int(entries[0]),
                                    'oword': oword, 
                                    'word': normalize(oword) if orig_word else oword.lower(),
                                    'cword': cword,
                                    'olemma': olemma,
                                    'lemma': olemma.lower(),
                                    'clemma': olemma.lower(), # for inflection, normally the same as lemma, only different in korean
                                    'upos': entries[3],
                                    'xpos': entries[4],
                                    'morph': morph,
                                    'morphstr': '|'.join(morph) if morph else '_',
                                    'omorphstr': entries[5],
                                    'hid': int(entries[6]),
                                    'label': entries[7],
                                    'lin' : lin,
                                    'original_id': original_id,
                                    'ids': ids,
                                    'deps': [],
                                    'lost': [],
                                    'domain': [],
                                    'linearized_domain': [], # T1 prediction
                                    'generated_domain': [], # T2 prediction
                                    'gold_linearized_domain': [],
                                    'phead': None, # for parser prediction
                                    'phid': None # for parser prediction
                                    })
                        if convert_lemma:
                            token.convert_lemma_morph()
                        token.get_diff()
                        if token['label'] == '<LOST>' and skip_lost:
                            sent.lost.append(token)
                        # elif token['lemma'] != '_': # ignore empty nodes
                        else:
                            sent.add_token(token)
            elif len(sent.tokens) > 1:
                if not simple:
                    sent.complete()
                count += 1
                yield sent
                sent = Sentence()
                if first and count >= first:
                    break

# default write ud format as output
def write_conllu(filename, sents, ud=True, use_morphstr=False, header=True):
    with open(filename, 'w') as out:
        sent_id = 0
        for sent in sents:
            sent_id += 1
            out_tokens = sent.get_output_tokens()
            if header:
                text = ' '.join((t['oword'] if t['oword'] != '_' else t['olemma']) for t in out_tokens if not t.get('contracted', False))
                line = f"# sent_id = {sent_id}\n# text = {text}\n"
            else:
                line = ''
            for t in out_tokens:
                if use_morphstr:
                    morphstr = t['omorphstr']
                else:
                    # morphstr = t['morphstr']
                    morphstr = '_' if t['morph'] == [] else \
                            '|'.join(m for m in sorted(t['morph'], key=str.swapcase))
                hid = t['phid'] if t.get('phid', None) is not None else t['hid'] # use predicted head if available
                if ud:
                    line += f"{t['tid']}\t{t['oword']}\t{t['olemma']}\t{t['upos']}\t{t['xpos']}\t{morphstr}\t" \
                            f"{hid}\t{t['label']}\t{t['cword']}\t{t['original_id'] or '_'}\n" 
                else:
                    line += f"{t['tid']}\t{t['olemma']}\t{t['oword']}\t{t['upos']}\t{t['xpos']}\t{morphstr}\t" \
                            f"{hid}\t{t['label']}\t{t['cword']}\t{t['original_id'] or '_'}\n" 
            line += '\n'
            out.write(line)

def write_txt(filename, sents):
    with open(filename, 'w') as out:
        sent_id = 0
        for sent in sents:
            sent_id += 1
            tokens = sent['gold_linearized_tokens'] or sent['gold_generated_tokens'] or sent.get_tokens()
            text = ' '.join(t['oword'] for t in tokens if t['oword'] and t.not_empty())
            line = f"# sent_id = {sent_id}\n# text = {text}\n\n"
            out.write(line)


def write_nbest(filename, sents, key='lemma'):
    # for the moment, always use the gold word if the inflected word is expected
    # TODO for later, store the inflected word for each linearization hypothesis
    assert key in ['lemma', 'word', 'oword']
    with open(filename, 'w') as out:
        sent_id = 0
        for sent in sents:
            sent_id += 1
            line = f"# sent_id = {sent_id}\n"
            # tokens = sent['gold_linearized_tokens'] or sent['gold_generated_tokens'] or sent.get_tokens()
            for tokens in sent['nbest_linearized_tokens']:
                text = ' '.join(t[key] for t in tokens if t[key] and t.not_empty())
                line += text+'\n'
            line += '\n'
            out.write(line)


def iterate_batch(train_sents, extra_sents, size, ratio):
    batch = []
    while True:
        for i in range(size):
            if not extra_sents or i % (ratio + 1) == 0:
                batch.append(next(train_sents))
            else:
                batch.append(next(extra_sents))
        random.shuffle(batch)
        yield batch
        batch = []

# def iterate_batch(train_sents, extra_sents, size, ratio):
#     tg = iterate(train_sents)
#     eg = iterate(extra_sents)
    
#     batch = []
#     while True:
#         for i in range(size):
#             if i % (ratio + 1) == 0 or not extra_sents:
#                 batch.append(next(tg))
#             else:
#                 batch.append(next(eg))
#         random.shuffle(batch)
#         yield batch
#         batch = []


def iterate(sents):
    while True:
        random.shuffle(sents)
        yield from sents

def iterate_sents(sents, extra_sents=[], ratio=1):
    tg = iterate(sents)
    eg = iterate(extra_sents)

    while True:
        yield next(tg)
        if extra_sents:
            for i in range(ratio):
                yield next(eg)

# flatten the linearized domain
def flatten(token, key='linearized_domain'):
    assert key in ['linearized_domain', 'generated_domain', 'gold_linearized_domain', 'gold_generated_domain']
    if token['tid'] is None:
        return [token] # generated tokens
    else:
        return sum([(flatten(tk, key) if (tk is not token) else ([tk] if token['tid'] != 0 and token.not_empty() else [])) \
                 for tk in token[key]], [])

def get_edit_diff(lemma, word):
    lemma, word = lemma.lower(), word.lower()
    if lemma == word:
        return '='
    diff = ''
    prev = ''
    for (tp, bl, el, bw, ew) in Levenshtein.opcodes(lemma, word):
        # print(tp, bl, el, bw, ew)
        if tp == 'equal':
            diff += '✓' * (el-bl)
        elif tp == 'delete':
            diff += '✗' * (el-bl)
        elif tp == 'insert':
            diff += word[bw:ew]
        elif tp == 'replace':
            if prev == 'delete':
                # diff += lemma[bl:el]
                diff += '✗' * (el-bl)
                diff += word[bw:ew]
            elif prev == 'insert':
                diff += word[bw:ew]
                diff += '✗' * (el-bl)
            # either prev == equal or no prev
            else:
                diff += '✗' * (el-bl)
                diff += word[bw:ew]
        prev = tp
    return diff

def get_word_from_edit_diff(lemma, diff):
    try:
        if diff == '=':
            return lemma
        word = ''
        i = 0
        for d in diff:
            if d == '✗':
                i += 1
            elif d == '✓':
                word += lemma[i]
                i += 1
            else:
                word += d
        return word
    except:
        return lemma

if __name__ == '__main__':
    # simple test 
    sents = list(read_conllu('data/T1-dev/es_gsd-ud-dev.conllu'))
    # sents = read_conllu('data/T2-dev/es_gsd-ud-dev_DEEP.conllu')
    write_conllu('tmp/pred/es_gsd-ud-dev.conllu', sents)