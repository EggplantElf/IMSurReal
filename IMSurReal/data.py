from collections import defaultdict
from itertools import repeat
import random

class Token(dict):
    def __init__(self, entries):
        for k, v in entries.items():
            self[k] = v

    def __repr__(self):
        # return f"<{self['tid']}, {self['olemma']}, {self['hid']}>"
        return f"{self['lemma']}<{self['tid']}>({self['original_id']})"

    def not_empty(self):
        return self['lemma'] != '_' or self['upos'] != 'PRON'

class Root(Token):
    def __init__(self, entries):
        super().__init__(entries)

class Sentence(dict):
    def __init__(self):
        self.meta = ''
        self['gold_contracted_tokens'] = [] # contracted token from ud file
        self['gold_contracted_lines'] = {} # contracted token from ud file
        self.lost = [] # gold lost token in T2
        self.tokens = [Root({'tid': 0,
                            'oword': '<ROOT>',
                            'olemma': '<ROOT>',
                            'lemma': '<root>',
                            'clemma': '<root>',
                            'cword': '', # word form after contraction
                            'upos': '<ROOT>',
                            'hid': -1,
                            'head': None,
                            'original_id': 0,
                            'deps': [],
                            'lost': [],
                            'gold_linearized_domain': [],
                            'linearized_domain': [],
                            'domain': []})]
        self.root = self.tokens[0]
        self.is_projective = True
        self.root['domain'].append(self.root) # root is in its own domain
        self['generated_tokens'] = []
        self['linearized_tokens'] = []      # obtained after running the linearization
                                            # predicted ordered tokens (excluding root)
        self['contracted_tokens'] = [] # add contracted token (e.g. 'zu', 'dem' -> 'zum') in the list,
                                            # the tid of the contracted token is '2-3' if tid of 'zu' is 2 and tid of 'dem' is 3
                                            # and the compoments 'zu' and 'dem' are marked with 'contracted' = True,
                                            # so that they won't appear in the output text, but still appear in the lines
        self['gold_linearized_tokens'] = []
        self['gold_generated_tokens'] = []

    def __repr__(self):
        return ' '.join([f"{t['lemma']}({t['tid']})" for t in self.tokens[1:]])

    def get_tokens(self, include_empty = True):
        return [t for t in self.tokens[1:] if include_empty or t.not_empty()]

    def add_token(self, token):
        self.tokens.append(token)

    def complete(self):
        # whatever needed to do after reading the sentence` 
        # e.g. check for projectivity

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

        # gather contracted tokens (only appear in UD file, thus gold tid)
        for line in self['gold_contracted_lines'].values():
            items = line.split('\t')
            word = items[1].lower()
            ids = items[0].split('-')
            b, e = int(ids[0]), int(ids[1])
            tokens = [self.tokens[i] for i in range(b, e+1)]
            self['gold_contracted_tokens'].append((word, tokens))
            tokens[0]['cword'] = word
            for t in tokens[1:]:
                t['cword'] = ''


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
                self['linearized_tokens'] or \
                self.tokens[1:]

# default read non-ud format as input, which means lemma and word are swapped 
def read_conllu(filename, ud=False, skip_lost=True, first = None):
    sents = []
    sent = Sentence()
    for line in open(filename):
        if line.strip():
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
                    # original_id = int(entries[0])  # used in both T1 and T2 to indicate the original tid in UD, 
                    original_id = None  # used in both T1 and T2 to indicate the original tid in UD, 
                                        # only in train set
                    ids = []            # used in T2 to indicate all corresponding tokens (head and deleted children) in T1,
                                        # only in train set

                    oword = entries[1] if (ud or entries[2] == '_') else entries[2]
                    olemma = entries[2] if (ud and entries[2] != '_') else entries[1]
                    cword = oword.lower() if (ud or entries[8] == '_') else entries[8]
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
                                  'word': oword.lower(),
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
                                   })
                    if token['label'] == '<LOST>' and skip_lost:
                        sent.lost.append(token)
                    # elif token['lemma'] != '_': # ignore empty nodes
                    else:
                        sent.add_token(token)
        elif len(sent.tokens) > 1:
            sent.complete()
            sents.append(sent)
            sent = Sentence()
            if first and len(sents) >= first:
                break
    return sents

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
                if ud:
                    line += f"{t['tid']}\t{t['oword']}\t{t['olemma']}\t{t['upos']}\t{t['xpos']}\t{morphstr}\t" \
                            f"{t['hid']}\t{t['label']}\t{t['cword']}\t{t['original_id'] or '_'}\n" 
                else:
                    line += f"{t['tid']}\t{t['olemma']}\t{t['oword']}\t{t['upos']}\t{t['xpos']}\t{morphstr}\t" \
                            f"{t['hid']}\t{t['label']}\t{t['cword']}\t{t['original_id'] or '_'}\n" 
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


def iterate(sents, max_step=100000):
    step = 0
    while step < max_step:
        random.shuffle(sents)
        for sent in sents:
            step += 1
            yield step, sent


if __name__ == '__main__':
    # simple test 
    sents = read_conllu('data/T1-dev/es_gsd-ud-dev.conllu')
    # sents = read_conllu('data/T2-dev/es_gsd-ud-dev_DEEP.conllu')
    write_conllu('tmp/pred/es_gsd-ud-dev.conllu', sents)