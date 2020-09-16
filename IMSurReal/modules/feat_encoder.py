import dynet as dy
import dynet_modules as dm
import numpy as np
import gzip, pickle
from collections import defaultdict
from utils import *

class FeatEncoder(Encoder):
    def __init__(self, args, model, train_sents=None):
        super().__init__(args, model)

        # train mode
        if train_sents:
            self.get_maps(train_sents)
        else:
            self.load_maps()

        # create parameters
        if 'word' in self.args.features:
            self.word_emb = self.model.add_lookup_parameters((len(self.word_map), self.args.hid_dim))
        if 'lemma' in self.args.features:
            self.lemma_emb = self.model.add_lookup_parameters((len(self.lemma_map), self.args.hid_dim))
        if 'upos' in self.args.features:
            self.upos_emb = self.model.add_lookup_parameters((len(self.upos_map), self.args.hid_dim))
        if 'xpos' in self.args.features:
            self.xpos_emb = self.model.add_lookup_parameters((len(self.xpos_map), self.args.hid_dim))
        if 'morph' in self.args.features:
            self.morph_emb = self.model.add_lookup_parameters((len(self.morph_map), self.args.hid_dim))
            self.morph_lstm_encoder = dy.VanillaLSTMBuilder(1, self.args.hid_dim, self.args.hid_dim, self.model)
        if 'label' in self.args.features:
            self.label_emb = self.model.add_lookup_parameters((len(self.label_map), self.args.hid_dim))

        if 'char_lstm' in self.args.features or 'inf' in self.args.tasks or 'con' in self.args.tasks:
            self.char_emb = self.model.add_lookup_parameters((len(self.char_map), self.args.hid_dim))
            if 'char_lstm' in self.args.features:
                self.char_lstm_f_encoder = dy.VanillaLSTMBuilder(1, self.args.hid_dim, self.args.hid_dim/2, self.model)
                self.char_lstm_b_encoder = dy.VanillaLSTMBuilder(1, self.args.hid_dim, self.args.hid_dim/2, self.model)
        self.log(f'Initialized <{self.__class__.__name__}>, params = {self.model.parameter_count()}')


    def load_maps(self):
        print('load maps')
        with gzip.open(self.args.model_file+'.maps.gz','rb') as stream:
            self.char_map = pickle.load(stream)
            self.word_map = pickle.load(stream)
            self.lemma_map = pickle.load(stream)
            self.morph_map = pickle.load(stream)
            self.upos_map = pickle.load(stream)
            self.xpos_map = pickle.load(stream)
            self.label_map = pickle.load(stream)
            self.lost_map = pickle.load(stream)
            self.inf_rules = pickle.load(stream)



    def get_maps(self, sents):
        self.args.num_train_sents = 0
        print('get maps')
        self.char_map = {'<#c?>': 0, '<$>': 1}
        self.word_map = {'<#w?>': 0, '<#W?>': 1}
        self.lemma_map = {'<#l?>': 0}
        self.morph_map = {'<#m?>': 0}
        self.upos_map = {'<#u?>': 0}
        self.xpos_map = {'<#x?>': 0}
        self.label_map = {'<#l?>': 0}
        self.lost_map = {('<$$$>', '<$$$>', '<$$$>'): 0}

        self.char_freq = defaultdict(int)
        self.word_freq = defaultdict(int)
        self.lemma_freq = defaultdict(int)
        self.morph_freq = defaultdict(int)
        self.upos_freq = defaultdict(int)
        self.xpos_freq = defaultdict(int)
        self.label_freq = defaultdict(int)
        self.lost_freq = defaultdict(int)
        self.upos2sigs = defaultdict(set)

        # for extracting inflection dictionary
        inf_freq = defaultdict(lambda: defaultdict(int))
        self.inf_rules = {}

        # count the frequency of each feature
        for sent in sents:
            self.args.num_train_sents += 1
            for token in sent.get_tokens():
                for c in token['word'] + token['lemma']:
                    self.char_freq[c] += 1
                for m in token['morph']:
                    self.morph_freq[m] += 1
                    # for char seq2seq
                    self.char_freq[m] += 1 
                self.word_freq[token['word']] += 1
                self.lemma_freq[token['lemma']] += 1
                self.upos_freq[token['upos']] += 1
                # for morph lstm
                self.morph_freq[token['upos']] += 1
                # for char lstm
                self.char_freq[token['upos']] += 1 

                self.xpos_freq[token['xpos']] += 1
                self.label_freq[token['label']] += 1
                # token['diff'] = get_edit_diff(token['clemma'], token['word'])
                if 'inf' in self.args.tasks:
                    inf_freq[f"{token['clemma']}-{token['upos']}-({'|'.join(token['morph'])})"][token['word']] += 1

            # extract lost tokens for T2
            for token in sent.lost:
                self.lost_freq[signature(token)] += 1
                self.upos2sigs[token['upos']].add(signature(token))

        # extract inflection rules
        if 'inf' in self.args.tasks:
            min_freq = 5
            min_certainty = 0.99
            for k, v in sorted(inf_freq.items(), key=lambda x: -sum(x[1].values())):
                if sum(v.values()) > min_freq:
                    w, c = max(v.items(), key=lambda x:x[1])
                    if c > sum(v.values()) * min_certainty: 
                        self.inf_rules[k] = w
            self.log(f'total inflection rules: {len(self.inf_rules)}')

        self.log(f'LEMMA={len(self.lemma_freq)}') # count lemma before limit
        # limit the vocabulary
        self.word_freq = dict(sorted(self.word_freq.items(), key=lambda x: -x[1])[:self.args.max_vocab])
        self.lemma_freq = dict(sorted(self.lemma_freq.items(), key=lambda x: -x[1])[:self.args.max_vocab])

        # only take features with certain minimum frequency (1 for now)
        self.char_map.update({k:i+2 for i, k in enumerate(k for k, v in self.char_freq.items() if v > 0)})
        self.word_map.update({k:i+2 for i, k in enumerate(k for k, v in self.word_freq.items() if v > 0)})
        self.lemma_map.update({k:i+1 for i, k in enumerate(k for k, v in self.lemma_freq.items() if v > 0)})
        self.morph_map.update({k:i+1 for i, k in enumerate(k for k, v in self.morph_freq.items() if v > 0)})
        self.upos_map.update({k:i+1 for i, k in enumerate(k for k, v in self.upos_freq.items() if v > 0)})
        self.xpos_map.update({k:i+1 for i, k in enumerate(k for k, v in self.xpos_freq.items() if v > 0)})
        self.label_map.update({k:i+1 for i, k in enumerate(k for k, v in self.label_freq.items() if v > 0)})
        self.lost_map.update({k:i+1 for i, k in enumerate(k for k, v in self.lost_freq.items() if v > 0)})

        self.word_drop_list = set([l for l, v in self.word_freq.items() if v < 5])
        self.lemma_drop_list = set([l for l, v in self.lemma_freq.items() if v < 5])


        # automatically use xpos if the size is much larger than upos but not too big
        # if self.args.auto_xpos and 'xpos' not in self.args.features \
        if 'xpos' not in self.args.features and not self.args.no_xpos \
                and 500 > len(self.xpos_map) > len(self.upos_map) * 2:
            self.log('automatically add xpos as feature')
            self.args.features.append('xpos')


        # token represntation dimension is the same as all hidden dimensions, 
        # since all the feature vectors are summed instead of concatenated
        self.args.token_dim = self.args.hid_dim
        self.args.char_dim = self.args.hid_dim
        self.args.morph_dim = self.args.hid_dim

        # target
        self.chars = list(self.char_map.keys())
        self.losts = list(self.lost_map.keys()) 

        # map infrequent lost tokens to the most similar one
        for upos in self.upos2sigs.keys():
            self.upos2sigs[upos] = sorted([sig for sig in self.upos2sigs[upos] if sig in self.lost_map], 
                                          key=lambda x: -self.lost_freq[x])
        for sent in sents:
            for token in sent.lost:
                sig = signature(token)
                if sig not in self.lost_map:
                    cands = self.upos2sigs[upos] or self.losts[1:]
                    similar_sig = min(cands, key=lambda x: Levenshtein.distance(x[0], sig[0]))

                    token['lemma'] = similar_sig[0]
                    token['upos'] = similar_sig[1]
                    token['morph'] = similar_sig[2].split('|')


        self.log(f'raw counts: char={len(self.char_freq)}, word={len(self.word_freq)}, lemma={len(self.lemma_freq)}, morph={len(self.morph_freq)}, '\
                 f'upos={len(self.upos_freq)}, xpos={len(self.xpos_freq)}, label={len(self.label_freq)}, lost={len(self.lost_freq)}')
        self.log(f'map sizes: char={len(self.char_map)}, word={len(self.word_map)}, lemma={len(self.lemma_map)}, morph={len(self.morph_map)}, '\
                 f'upos={len(self.upos_map)}, xpos={len(self.xpos_map)}, label={len(self.label_map)}, lost={len(self.lost_map)}')

        # save the maps
        with gzip.open(self.args.model_file+'.maps.gz', 'wb') as stream:
            pickle.dump(self.char_map, stream, -1)
            pickle.dump(self.word_map, stream, -1)
            pickle.dump(self.lemma_map, stream, -1)
            pickle.dump(self.morph_map, stream, -1)
            pickle.dump(self.upos_map, stream, -1)
            pickle.dump(self.xpos_map, stream, -1)
            pickle.dump(self.label_map, stream, -1)
            pickle.dump(self.lost_map, stream, -1)
            pickle.dump(self.inf_rules, stream, -1)



    def encode(self, sent, train_mode=False):
        # encode the root
        # sent.root.vecs['feat'] = self.special[0]

        for token in sent.get_tokens():
            vecs = []
            if 'word' in self.args.features:
                if train_mode and np.random.random() < 0.01 and token['word'] in self.word_drop_list:
                    word_idx = int(token['word'][0].isupper())
                else:
                    word_idx = self.word_map.get(token['word'], int(token['word'][0].isupper()))

                word_vec = self.word_emb[word_idx] if word_idx else dy.inputVector(np.zeros(self.args.hid_dim))
                vecs.append(word_vec)
            if 'lemma' in self.args.features:
                lemma_idx = 0 if train_mode and np.random.random() < 0.1 and token['lemma'] in self.lemma_drop_list\
                            else self.lemma_map.get(token['lemma'], 0)
                lemma_vec = self.lemma_emb[lemma_idx] if lemma_idx else dy.inputVector(np.zeros(self.args.hid_dim))
                vecs.append(lemma_vec)
            if 'upos' in self.args.features:
                upos_vec = self.upos_emb[self.upos_map.get(token['upos'], 0)]
                vecs.append(upos_vec)
            if 'xpos' in self.args.features:
                vecs.append(self.xpos_emb[self.xpos_map.get(token['xpos'], 0)])
            if 'label' in self.args.features:
                vecs.append(self.label_emb[self.label_map.get(token['label'], 0)])
            if 'char_lstm' in self.args.features:
                char_vecs = [self.char_emb[self.char_map.get(c, 0)] for c in token['clemma']]
                f_vecs = self.char_lstm_f_encoder.initial_state().transduce(char_vecs)
                b_vecs = self.char_lstm_b_encoder.initial_state().transduce(reversed(char_vecs))
                char_vec = dy.concatenate([f_vecs[-1], b_vecs[-1]])
                vecs.append(char_vec)
            # if 'morph' in self.args.features and 'lemma' in self.args.features and 'upos' in self.args.features:
            if 'morph' in self.args.features:
                morph_items = ([token['upos']] if 'upos' in self.args.features else ['<#m?>']) + token['morph'] 
                morph_input = [self.morph_emb[self.morph_map.get(m, 0)] for m in morph_items]
                morph_vec = self.morph_lstm_encoder.initial_state().transduce(morph_input)[-1]
                vecs.append(morph_vec)
            # token.vecs['feat'] = dy.concatenate(vecs)
            # token.vecs['feat'] = sum(vecs)
            token.vecs['feat'] = dy.dropout(sum(vecs), self.args.dropout) if train_mode else sum(vecs)


