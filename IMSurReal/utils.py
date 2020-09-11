import sys
import dynet_modules as dm
from data import Token, flatten
import dynet as dy
import numpy as np
import nltk.translate.bleu_score as bs
import gzip, pickle
from copy import copy


class Module(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model.add_subcollection()
        self.frozen = False

    def log(self, msg):
        if self.args.mode == 'train' and self.args.model_file:
            with open(self.args.model_file+'.log', 'a') as f:
                f.write(str(msg)+'\n')
        print(msg)

    def set_freeze(self, freeze=True):
        self.frozen = freeze
        print(f"{'Freeze' if freeze else 'Unfreeze'} <{self.__class__.__name__}>")
        for p in self.model.parameters_list() + self.model.lookup_parameters_list():
            p.set_updated(not freeze)

    def l2_norm(self):
        return self.l2_norm_params() + self.l2_norm_lookups()

    def l2_norm_params(self):
        return sum(dy.sum_elems(p**2) for p in self.model.parameters_list())

    def l2_norm_lookups(self):
        return sum(dy.sum_elems(p**2) for p in self.model.lookup_parameters_list())


class Encoder(Module):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.prev = None

    def encode(self, sent):
        print('encode() not implemented!')
        exit()

class Decoder(Module):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.prev = None

    def encode(self, sent):
        pass
        
    def train_one_step(self, sent):
        print('train_one_step() not implemented!')
        exit()


def traverse_topdown(h, pred=False):
    yield h
    for d in (h['pdeps'] if pred else h['deps']):
        if d.not_empty():
            yield from traverse_topdown(d, pred)

def traverse_bottomup(h, pred=False):
    for d in (h['pdeps'] if pred else h['deps']):
        if d.not_empty():
            yield from traverse_bottomup(d, pred)
    yield h


def sent_bleu(gold_seq, pred_seq, gkey='lemma', pkey='lemma'):
    chencherry = bs.SmoothingFunction()
    gs = [[t[gkey] for t in gold_seq]]
    ps = [t[pkey] for t in pred_seq]
    return bs.sentence_bleu(gs, ps, smoothing_function=chencherry.method2)


def eval_all(gold_seqs, pred_seqs, gkey='lemma', pkey='lemma'):
    all_ref = [ [[(token[gkey] if token['label'] != '<LOST>'else f"({token[gkey]})") for token in seq]] for seq in gold_seqs]
    all_hyp = [ [(token[pkey] if token['label'] != '<LOST>' else f"({token[pkey]})") for token in seq] for seq in pred_seqs]
    chencherry = bs.SmoothingFunction()
    bleu = bs.corpus_bleu(all_ref, all_hyp, smoothing_function=chencherry.method2)
    return bleu

def text_bleu(gold_txts, pred_txts):
    all_ref = [[txt.lower().split()] for txt in gold_txts]
    all_hyp = [txt.lower().split() for txt in pred_txts]
    chencherry = bs.SmoothingFunction()
    bleu = bs.corpus_bleu(all_ref, all_hyp, smoothing_function=chencherry.method2)
    return bleu


def capitalize(tokens, ignore_lemma_case=False):
    for t in tokens:
        # capitalize proper noun 
        if t['upos'] == 'PROPN':
            t['oword'] = t['oword'].capitalize()

        # follow the case of the lemma
        if not ignore_lemma_case:
            # copy lemma
            if t['olemma'].lower() == t['oword'].lower():
                t['oword'] = t['olemma']
            # all upper
            elif t['olemma'].isupper():
                t['oword'] = t['oword'].upper()
            # first upper
            elif t['olemma'][0].isupper():
                t['oword'] = t['oword'].capitalize()

    # capitalize the first (non-punct) token unless if it is lower
    words = [t for t in tokens if t['upos'] != 'PUNCT']
    if words and words[0]['oword'] and not words[0]['oword'][0].isupper():
    # if words and not words[0]['oword'][0].isupper():
        words[0]['oword'] = words[0]['oword'].capitalize()

def signature(t):
    # unique signature of the lost token
    return (t['lemma'], t['upos'], '|'.join(t['morph']) if t['morph'] else '_')

def inverse_num(tokens):
    return sum((t['original_id'] - i) for (i, t) in enumerate(tokens, 1) if t['original_id'] > i)

def reorder(sent, key='linearized_tokens'):
    assert key in ['linearized_tokens', 'generated_tokens', 'sorted_tokens']
    new_tokens = []
    for t in sent[key]:
        if t['tid'] is None:
            t['tid'] = len(sent.tokens+new_tokens)
            new_tokens.append(t)
    mapping = {t['tid']: i for i, t in enumerate(sent[key], 1)}
    for t in sent[key]:
        t['tid'] = mapping[t['tid']] # new id
        t['hid'] = mapping.get(t['hid'], 0) # new head


def sum_vecs(sent, out_key, in_keys):
    for token in sent.tokens:
        token.vecs[out_key] = sum(token.vecs[in_key] for in_key in in_keys if in_key in token.vecs)



