import sys
from data import read_conllu, write_conllu, iterate, write_txt
from utils import *
import dynet as dy
import dynet_modules as dm
import numpy as np
from argparse import ArgumentParser
import gzip, pickle
from time import time 
import re
from collections import defaultdict
from itertools import combinations
import Levenshtein

class Realization(object):
    """
    linearization and inflection with almost the same tree-based encoder,
    and an additional bilstm for the inflection, given the sentence order
    """
    def __init__(self, args):
        self.args = args
        self.args.stats = defaultdict(int)
        if args.mode == 'train':
            self.args.tasks = re.split(',|\+', self.args.tasks)
            self.args.features = re.split(',|\+', self.args.features)
            self.args.tree_vecs = re.split(',|\+', self.args.tree_vecs)
            self.args.lin_decoders = re.split(',|\+', self.args.lin_decoders)

            self.log(self.args.tasks)
            self.log(self.args.features)
            self.log(self.args.tree_vecs)
            self.log(self.args.lin_decoders)

            skip_lost = ('inf' not in  self.args.tasks and 'con' not in self.args.tasks)
            self.train_sents = read_conllu(self.args.train_file, False, skip_lost)
            self.dev_sents = read_conllu(self.args.dev_file, False, skip_lost)
            self.test_sents = read_conllu(self.args.input_file, False, skip_lost) if self.args.input_file else []

            # deal with korean lemma
            if self.args.lemmatize:
                self.log('lemmatize')
                convert_lemma_morph(self.train_sents + self.dev_sents + self.test_sents)

            self.get_maps(self.train_sents)

            self.log(f'train sents: {len(self.train_sents)}')
            self.log(f'dev sents: {len(self.dev_sents)}')
            self.log(f'test sents: {len(self.test_sents)}')

            # check oov 
            oov = total = 0
            for sent in self.dev_sents:
                for t in sent.get_tokens():
                    total += 1
                    if t['lemma'] not in self.lemma_map:
                        oov += 1
            self.log(f'OOV: {oov} / {total} = {100*oov/total:.2f}')
            # exit()
        else:
            self.load_maps()
        self.log(self.args)

        self.model = dy.Model()

        # encode tokens (including context)
        self.lemma_emb = self.model.add_lookup_parameters((len(self.lemma_map), self.args.lemma_dim))
        self.char_emb = self.model.add_lookup_parameters((len(self.char_map), self.args.char_dim))
        self.upos_emb = self.model.add_lookup_parameters((len(self.upos_map), self.args.upos_dim))
        self.xpos_emb = self.model.add_lookup_parameters((len(self.xpos_map), self.args.xpos_dim))
        self.morph_emb = self.model.add_lookup_parameters((len(self.morph_map), self.args.morph_dim))
        self.label_emb = self.model.add_lookup_parameters((len(self.label_map), self.args.label_dim))
        self.morphstr_emb = self.model.add_lookup_parameters((len(self.morphstr_map), self.args.morphstr_dim))


        self.char_lstm_f_encoder = dy.VanillaLSTMBuilder(1, self.args.char_dim, self.args.lemma_dim/2, self.model)
        self.char_lstm_b_encoder = dy.VanillaLSTMBuilder(1, self.args.char_dim, self.args.lemma_dim/2, self.model)
        self.morph_lstm_encoder = dy.VanillaLSTMBuilder(1, self.args.morph_dim, self.args.morph_dim, self.model)
        self.head_chain_encoder = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, self.model)

        self.root_vec = self.model.add_parameters(self.args.token_dim)
        self.dummy_morph_vec = self.model.add_parameters(self.args.morph_dim)
        self.sum_w = self.model.add_parameters((self.args.token_dim, self.args.full_dim),
                init=dm.orthonormal_initializer(self.args.token_dim, self.args.full_dim))
        self.sum_b = self.model.add_parameters(self.args.token_dim)

        self.tree_lstm = dm.TreeLSTM(self.model, self.args.token_dim, self.args.tree_lstm)


        # for linearization
        if 'lin' in self.args.tasks:
            if 'l2r' in self.args.lin_decoders:
                self.l2r_linearizer = L2RLinearizer(self.model, self.args) 

            if 'r2l' in self.args.lin_decoders:
                self.r2l_linearizer = R2LLinearizer(self.model, self.args) 

            if 'h2d' in self.args.lin_decoders:
                self.h2d_linearizer = H2DLinearizer(self.model, self.args) 

            if self.args.ranker:
                self.ranker = SentRanker(self.model, self.args)

        if 'gen' in self.args.tasks:
            self.h2d_generator = H2DGenerator(self.model, self.args, self.lost_map) 

        if 'inf' in self.args.tasks:
            self.token_seq_encoder = dy.BiRNNBuilder(1, self.args.token_dim, self.args.token_dim, self.model, dy.VanillaLSTMBuilder)

            self.inf_rules = {}
            if not self.args.no_rules:
                if self.args.mode == 'train':
                    self.extract_rules(self.train_sents, 5, 0.99)
                    self.eval_rules(self.dev_sents)
                else:
                    self.load_rules()

            self.inf_decoder = EditSeq2SeqDecoder(self.inf_rules, self.model, self.args, self.char_map, self.char_emb)

        if 'con' in self.args.tasks:
            self.token_seq_encoder = dy.BiRNNBuilder(1, self.args.token_dim, self.args.token_dim, self.model, dy.VanillaLSTMBuilder)
            self.con_decoder = ContractDecoder(self.model, self.args, self.char_map, self.char_emb)



        self.log(f'Total params: {self.model.parameter_count()}')

        self.trainer = dy.AdamTrainer(self.model)
        self.trainer.set_clip_threshold(1)

        if args.mode != 'train':
            self.load_model()


    def log(self, msg):
        if self.args.mode == 'train' and self.args.model_file:
            with open(self.args.model_file+'.log', 'a') as f:
                f.write(str(msg)+'\n')
        print(msg)

    def get_maps(self, sents):
        self.char_map = {'<#c?>': 0, '<$>': 1} # pad for cnn char encoder
        # self.oword_map = {'<#w?>': 0}
        self.lemma_map = {'<#l?>': 0}
        self.morph_map = {'<#m?>': 0}
        self.upos_map = {'<#u?>': 0}
        self.xpos_map = {'<#x?>': 0}
        self.label_map = {'<#l?>': 0}
        self.morphstr_map = {'<#ms?>': 0}
        self.lost_map = {('<$$$>', '<$$$>', '<$$$>'): 0}

        self.char_freq = defaultdict(int)
        # self.oword_freq = defaultdict(int)
        self.lemma_freq = defaultdict(int)
        self.morph_freq = defaultdict(int)
        self.upos_freq = defaultdict(int)
        self.xpos_freq = defaultdict(int)
        self.label_freq = defaultdict(int)
        self.morphstr_freq = defaultdict(int)
        self.lost_freq = defaultdict(int)
        self.upos2sigs = defaultdict(set)


        # count the frequency of each feature
        for sent in sents:
            for token in sent.get_tokens():
                for c in token['word'] + token['lemma']:
                    self.char_freq[c] += 1
                for m in token['morph']:
                    self.morph_freq[m] += 1
                    # for char seq2seq
                    self.char_freq[m] += 1 
                # self.oword_freq[token['oword']] += 1
                self.lemma_freq[token['lemma']] += 1
                self.upos_freq[token['upos']] += 1
                # for morph lstm
                self.morph_freq[token['upos']] += 1
                # for char lstm
                self.char_freq[token['upos']] += 1 

                self.xpos_freq[token['xpos']] += 1
                self.label_freq[token['label']] += 1
                self.morphstr_freq[token['morphstr']] += 1
                # token['diff'] = get_edit_diff(token['clemma'], token['cword'])
                token['diff'] = get_edit_diff(token['clemma'], token['word'])

            # extract lost tokens for T2
            for token in sent.lost:
                # self.lost_freq[token['upos']] += 1
                # self.lost_freq[token['lemma']] += 1
                # self.upos2sigs[token['upos']].add(token['lemma'])
                self.lost_freq[signature(token)] += 1
                self.upos2sigs[token['upos']].add(signature(token))


        # only take features with certain minimum frequency (1 for now)
        self.char_map.update({k:i+2 for i, k in enumerate(k for k, v in self.char_freq.items() if v > 0)})
        self.lemma_map.update({k:i+1 for i, k in enumerate(k for k, v in self.lemma_freq.items() if v > 0)})
        self.morph_map.update({k:i+1 for i, k in enumerate(k for k, v in self.morph_freq.items() if v > 0)})
        self.upos_map.update({k:i+1 for i, k in enumerate(k for k, v in self.upos_freq.items() if v > 0)})
        self.xpos_map.update({k:i+1 for i, k in enumerate(k for k, v in self.xpos_freq.items() if v > 0)})
        self.label_map.update({k:i+1 for i, k in enumerate(k for k, v in self.label_freq.items() if v > 0)})
        self.morphstr_map.update({k:i+1 for i, k in enumerate(k for k, v in self.morphstr_freq.items() if v > 0)})
        self.lost_map.update({k:i+1 for i, k in enumerate(k for k, v in self.lost_freq.items() if v > 0)})

        self.lemma_drop_list = [l for l, v in self.lemma_freq.items() if v < 5]

        # fixed dims
        self.args.char_dim = 64
        self.args.lemma_dim = 64
        self.args.morph_dim = 32
        self.args.upos_dim = 32
        self.args.xpos_dim = 32
        self.args.label_dim = 32
        self.args.morphstr_dim = 64

        # automatically use xpos if the size is much larger than upos but not too big
        if self.args.auto_xpos and 'xpos' not in self.args.features \
                and 500 > len(self.xpos_map) > len(self.upos_map) * 2:
            self.log('automatically add xpos as feature')
            self.args.features.append('xpos')


        self.args.token_dim = ('lemma' in self.args.features and self.args.lemma_dim) + \
                              ('char_lstm' in self.args.features and self.args.lemma_dim) + \
                              ('upos' in self.args.features and self.args.upos_dim) + \
                              ('xpos' in self.args.features and self.args.xpos_dim) + \
                              ('morph' in self.args.features and self.args.morph_dim) + \
                              ('label' in self.args.features and self.args.label_dim) + \
                              ('morphstr' in self.args.features and self.args.morphstr_dim)

        self.args.full_dim = (self.args.token_dim if 'self' in self.args.tree_vecs else 0) + \
                            (self.args.token_dim if 'head' in self.args.tree_vecs else 0) + \
                            (self.args.token_dim if 'deps' in self.args.tree_vecs else 0)

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


        self.log(f'raw counts: char={len(self.char_freq)}, lemma={len(self.lemma_freq)}, morph={len(self.morph_freq)}, morphstr={len(self.morphstr_freq)}, '\
                 f'upos={len(self.upos_freq)}, xpos={len(self.xpos_freq)}, label={len(self.label_freq)}, lost={len(self.lost_freq)}')
        self.log(f'map sizes: char={len(self.char_map)}, lemma={len(self.lemma_map)}, morph={len(self.morph_map)}, morphstr={len(self.morphstr_map)}, '\
                 f'upos={len(self.upos_map)}, xpos={len(self.xpos_map)}, label={len(self.label_map)}, lost={len(self.lost_map)}')
        self.log(f'emb dims: char={self.args.char_dim}, lemma={self.args.lemma_dim}, morph={self.args.morph_dim}, morphstr={self.args.morphstr_dim}, '\
                 f'upos={self.args.upos_dim}, xpos={self.args.xpos_dim}, label={self.args.label_dim}, all={self.args.token_dim}')

        # save the maps
        with gzip.open(self.args.model_file+'.gz', 'wb') as stream:
            pickle.dump(self.char_map, stream, -1)
            pickle.dump(self.lemma_map, stream, -1)
            pickle.dump(self.morph_map, stream, -1)
            pickle.dump(self.upos_map, stream, -1)
            pickle.dump(self.xpos_map, stream, -1)
            pickle.dump(self.label_map, stream, -1)
            pickle.dump(self.morphstr_map, stream, -1)
            pickle.dump(self.lost_map, stream, -1)
            pickle.dump(self.args, stream, -1)

    def load_maps(self):
        with gzip.open(self.args.model_file+'.gz','rb') as stream:
            self.char_map = pickle.load(stream)
            self.lemma_map = pickle.load(stream)
            self.morph_map = pickle.load(stream)
            self.upos_map = pickle.load(stream)
            self.xpos_map = pickle.load(stream)
            self.label_map = pickle.load(stream)
            self.morphstr_map = pickle.load(stream)
            self.lost_map = pickle.load(stream)

            # only use the new args related to files
            keep_args = self.args
            self.args = pickle.load(stream)
            self.args.mode = keep_args.mode
            self.args.model_file = keep_args.model_file
            self.args.input_file = keep_args.input_file            
            self.args.gold_file = keep_args.gold_file            
            self.args.pred_file = keep_args.pred_file

    def show_example_data(self, sent):
        features = [f for f in ['lemma', 'upos', 'xpos', 'label', 'morph'] if f in self.args.features]
        try:
            for t in sorted(sent.get_tokens(), key=lambda x: x['original_id']):
                self.log('\t'.join(str(t[feat]) for feat in features + (['word'] if 'inf' in self.args.tasks else [])))
        except:
            for t in sent.get_tokens():
                self.log('\t'.join(str(t[feat]) for feat in features + (['word'] if 'inf' in self.args.tasks else [])))


    def save_model(self):
        self.log('saving model')
        self.model.save(self.args.model_file)

    def load_model(self):
        self.model.populate(self.args.model_file)


    ##################################################
    # Encoding functions: 
    # common for linearization and inflection
    def encode_tokens(self, tokens, train_mode=False):
        for token in tokens:
            vecs = []
            if 'lemma' in self.args.features:
                lemma_idx = 0 if train_mode and np.random.random() < 0.01 and token['lemma'] in self.lemma_drop_list\
                                             else self.lemma_map.get(token['lemma'], 0)
                lemma_vec = self.lemma_emb[lemma_idx]
                vecs.append(lemma_vec)
            if 'upos' in self.args.features:
                upos_vec = self.upos_emb[self.upos_map.get(token['upos'], 0)]
                vecs.append(upos_vec)
            if 'xpos' in self.args.features:
                vecs.append(self.xpos_emb[self.xpos_map.get(token['xpos'], 0)])
            if 'label' in self.args.features:
                vecs.append(self.label_emb[self.label_map.get(token['label'], 0)])
            if 'morphstr' in self.args.features:
                vecs.append(self.morphstr_emb[self.morphstr_map.get(token['morphstr'], 0)])
            if 'char_lstm' in self.args.features:
                char_vecs = [self.char_emb[self.char_map.get(c, 0)] for c in token['lemma']]
                f_vecs = self.char_lstm_f_encoder.initial_state().transduce(char_vecs)
                b_vecs = self.char_lstm_b_encoder.initial_state().transduce(reversed(char_vecs))
                char_vec = dy.concatenate([f_vecs[-1], b_vecs[-1]])
                vecs.append(char_vec)
            if 'morph' in self.args.features and 'lemma' in self.args.features and 'upos' in self.args.features:
                morph_input = [self.morph_emb[self.morph_map.get(m, 0)] for m in ([token['upos']] + token['morph'])]
                morph_vec = self.morph_lstm_encoder.initial_state().transduce(morph_input)[-1]
                vecs.append(morph_vec)
            token['self_vec'] = dy.concatenate(vecs)


    def encode_head(self, token, head=None):
        head_state = head['head_state'] if head else self.head_chain_encoder.initial_state()
        if 'deps' in self.args.tree_vecs:
            # independent top-down pass
            if self.args.head_input == 'self_vec':
                token['head_state'] = head_state.add_input(token['self_vec'])
            # use bottom-up lstm output
            elif self.args.head_input == 'deps_vec':
                token['head_state'] = head_state.add_input(token['deps_vec'])
            # use bottom-up lstm hidden cell
            else:
                token['head_state'] = head_state.add_input(token['deps_mem'])
        else:
            token['head_state'] = head_state.add_input(token['self_vec'])
        token['head_vec'] = token['head_state'].output()
        for dep in token['deps']:
            self.encode_head(dep, token)

    def encode_deps(self, head):
        # propagate information bottom up 
        for dep in head['deps']:
            self.encode_deps(dep)

        if head['deps']:
            x = head['self_vec']
            hs = [dep['deps_vec'] for dep in head['deps']]
            cs = [dep['deps_mem'] for dep in head['deps']]
            head['deps_vec'], head['deps_mem'] = self.tree_lstm.state(x, hs, cs)
        else:
            head['deps_vec'], head['deps_mem'] = self.tree_lstm.state(head['self_vec'])

    def encode_sent(self, sent, train_mode=False):
        # encode token self
        sent.root['self_vec'] = sent.root['vec'] = self.root_vec
        self.encode_tokens(sent.tokens[1:], train_mode)

        self.encode_deps(sent.root)

        self.encode_head(sent.root) 

        for token in sent.get_tokens():
            vecs = []
            if 'self' in self.args.tree_vecs:
                vecs += [token['self_vec']]
            if 'head' in self.args.tree_vecs:
                vecs += [token['head_vec']]
            if 'deps' in self.args.tree_vecs:
                vecs += [token['deps_vec']] 

            token['vec'] = self.sum_b + self.sum_w * dy.concatenate(vecs)
            # residual
            if 'self' not in self.args.tree_vecs:
                token['vec'] += token['self_vec']

        # only for inflection!
        if ('inf' in self.args.tasks or 'con' in self.args.tasks) and not self.args.no_seq:
            tokens = sent['gold_linearized_tokens'] or sent['linearized_tokens'] or sent.get_tokens()
            vecs = self.token_seq_encoder.transduce([t['vec'] for t in tokens])
            for t, v in zip(tokens, vecs):
                t['vec'] += v



    def train(self):
        lin_time = lin_loss = lin_correct = lin_total = 0
        inf_time = inf_loss = inf_correct = inf_total = 0
        gen_time = gen_loss = gen_correct = gen_total = 0
        con_time = con_loss = con_correct = con_total = 0

        best_score = waited = 0

        for step, sent in iterate(self.train_sents):
            if 'lin' in self.args.tasks:
                res = self.train_linearization_one_step(sent)
                lin_time += res['time']
                lin_loss += res['loss']
                lin_total += res['total']
                lin_correct += res['correct']

            if 'gen' in self.args.tasks:
                res = self.train_generation_one_step(sent)
                gen_time += res['time']
                gen_loss += res['loss']
                gen_total += res['total']
                gen_correct += res['correct']

            if 'inf' in self.args.tasks:
                res = self.train_inflection_one_step(sent)                
                inf_time += res['time']
                inf_loss += res['loss']
                inf_total += res['total']
                inf_correct += res['correct']

            if 'con' in self.args.tasks:
                res = self.train_contraction_one_step(sent)                
                con_time += res['time']
                con_loss += res['loss']
                con_total += res['total']
                con_correct += res['correct']


            if step % self.args.eval_every == 0:
                t0 = time()
                res = self.predict(self.dev_sents[:500], True)
                print(res)
                if 'lin' in self.args.tasks:
                    self.log(f"[step={step}]\tlin_loss={lin_loss/self.args.eval_every:.2f}, "\
                        f"train_lin_acc={lin_correct}/{lin_total}={100*lin_correct/lin_total:.2f}, train_lin_time={lin_time:.1f}s")
                    self.log(f"dev_lin_bleu={100*res['lin_bleu']:.2f}, time={time()-t0:.1f}s")
                    if self.args.ranker:
                        self.log(f"dev_lin_random_bleu={100*res['lin_random_bleu']:.2f}, time={time()-t0:.1f}s")
                        self.log(f"dev_lin_oracle_bleu={100*res['lin_oracle_bleu']:.2f}, time={time()-t0:.1f}s")
                        self.log(f"dev_lin_ranked_bleu={100*res['lin_ranked_bleu']:.2f}, time={time()-t0:.1f}s")
                if 'gen' in self.args.tasks:
                    self.log(f"[step={step}]\tgen_loss={gen_loss/self.args.eval_every:.2f}, "\
                        f"train_gen_acc={gen_correct}/{gen_total}={100*gen_correct/gen_total:.2f}, train_gen_time={gen_time:.1f}s")
                    self.log(f"dev_gen_bleu={100*res['gen_bleu']:.2f}, time={time()-t0:.1f}s")
                if 'inf' in self.args.tasks:
                    self.log(f"[step={step}]\tinf_loss={inf_loss/self.args.eval_every:.2f}, "\
                        f"train_inf_acc={inf_correct}/{inf_total}={100*inf_correct/inf_total:.2f}, train_inf_time={inf_time:.1f}s")
                    self.log(f"dev_inf_acc={100*res['inf_acc']:.2f}, time={time()-t0:.1f}s")
                if 'con' in self.args.tasks:
                    self.log(f"[step={step}]\tcon_loss={con_loss/self.args.eval_every:.2f}, "\
                        f"train_con_acc={con_correct}/{con_total}={100*con_correct/con_total:.2f}, train_con_time={con_time:.1f}s")
                    self.log(f"dev_con_acc={100*res['con_acc']:.2f}, time={time()-t0:.1f}s")

                score = (res['lin_ranked_bleu'] if self.args.ranker else res['lin_bleu']) + res['gen_bleu'] + res['inf_acc'] + res['con_acc']
                if score > best_score:
                    best_score = score
                    self.save_model()
                    waited = 0
                # only start counting after finishing at least one iteration
                elif step > len(self.train_sents):
                    waited += 1
                    if waited > self.args.patience:
                        self.log('out of patience')
                        break

                lin_time = lin_loss = lin_correct = lin_total = 0
                inf_time = inf_loss = inf_correct = inf_total = 0
                gen_time = gen_loss = gen_correct = gen_total = 0


            if step > self.args.max_step:
                break

        self.load_model()
        self.log('FINAL DEV')
        res = self.predict(self.dev_sents, True)
        self.log(f"best_lin_bleu={100*res['lin_bleu']:.2f}")
        self.log(f"best_gen_bleu={100*res['gen_bleu']:.2f}")
        self.log(f"best_inf_acc={100*res['inf_acc']:.2f}")
        self.log(f"best_con_acc={100*res['con_acc']:.2f}")

        if self.test_sents:
            self.log('FINAL TEST')
            res = self.predict(self.test_sents, True)
            self.log(f"test_lin_bleu={100*res['lin_bleu']:.2f}")
            self.log(f"test_gen_bleu={100*res['gen_bleu']:.2f}")
            self.log(f"test_inf_acc={100*res['inf_acc']:.2f}")
            self.log(f"test_con_acc={100*res['con_acc']:.2f}")


    def predict(self, sents, evaluate=True):
        for sent in sents:
            dy.renew_cg()
            self.encode_sent(sent)
            # clear previous predictions
            sent['linearized_tokens'] = []
            sent['generated_tokens'] = []
            for token in sent.tokens: 
                token['linearized_domain'] = []
                token['generated_domain'] = []

            if 'lin' in self.args.tasks:
                self.linearize(sent)
                if not evaluate:
                    self.reorder(sent, 'linearized_tokens')

            if 'inf' in self.args.tasks:
                self.inflect(sent)

            if 'gen' in self.args.tasks:
                self.generate(sent)
                if not evaluate:
                    self.reorder(sent, 'generated_tokens')

            if 'con' in self.args.tasks:
                self.contract(sent)

        if evaluate:
            return self.evaluate(sents)


    def evaluate(self, sents):
        res = {'lin_bleu': 0, 'gen_bleu': 0, 'inf_bleu': 0, 'inf_acc': 0, 'con_acc': 0}

        if 'lin' in self.args.tasks:
            gold_seqs = [sent['gold_linearized_tokens'] for sent in sents]
            pred_seqs = [sent['linearized_tokens'] for sent in sents]
            res['lin_bleu'] = eval_all(gold_seqs, pred_seqs)

            if self.args.ranker:
                pred_seqs = [sent['random_linearized_tokens'] for sent in sents]
                res['lin_random_bleu'] = eval_all(gold_seqs, pred_seqs)

                pred_seqs = [sent['oracle_linearized_tokens'] for sent in sents]
                res['lin_oracle_bleu'] = eval_all(gold_seqs, pred_seqs)

                pred_seqs = [sent['ranked_linearized_tokens'] for sent in sents]
                res['lin_ranked_bleu'] = eval_all(gold_seqs, pred_seqs)

        if 'gen' in self.args.tasks:
            gold_seqs = [sent['gold_generated_tokens'] for sent in sents]
            pred_seqs = [sent['generated_tokens'] for sent in sents]
            res['gen_bleu'] = eval_all(gold_seqs, pred_seqs)

        if 'inf' in self.args.tasks:
            tokens = sents[0]['gold_linearized_tokens'] or sents[0]['gold_generated_tokens'] or sents[0].get_tokens()
            print(' '.join(t['word'] for t in tokens))
            print(' '.join(t['oword'] for t in tokens))
            tokens = sum([sent.get_tokens() for sent in sents], [])
            total = len(tokens)
            correct = sum(int(tk['oword'] == tk['word']) for tk in tokens)
            # for tk in tokens:
            #     # if tk['word'] != tk['cword'] and tk['oword'] != tk['cword']:
            #     if tk['oword'] != tk['cword']:
            #         print(tk['oword'], tk['cword'], tk['lemma'])
            res['inf_acc'] = correct / total
        if 'con' in self.args.tasks:
            gold_txts, pred_txts = [], []
            for sent in sents:
                tokens = sent['gold_linearized_tokens'] or sent['gold_generated_tokens'] or sent.get_tokens()
                gold_txts.append(' '.join([tk['cword'] for tk in tokens if tk['cword'] and tk.not_empty()]))
                pred_txts.append(' '.join([tk['oword'] for tk in tokens if tk['oword'] and tk.not_empty()]))

            n = 0
            for i in range(len(sents)):
                if any([(t['cword'].lower() != t['word']) for t in sents[i]['gold_linearized_tokens']]):
                    print(gold_txts[i])
                    print(pred_txts[i])
                n += 1
                if n > 5:
                    break
            res['con_acc'] = text_bleu(gold_txts, pred_txts)
        return res

    ##################################################
    # inflection functions

    def train_inflection_one_step(self, sent):
        t0 = time()
        dy.renew_cg()
        self.encode_sent(sent, True)
        res = self.inf_decoder.decode(sent, True)
        loss = res['loss']
        if loss:
            loss_value = loss.value()
            loss.backward()
            try:
                self.trainer.update()
            except:
                self.log('training error, load check point')
                self.load_model()
        else:
            loss_value = 0
        return {'time': time()-t0,
                'loss': loss_value,
                'total': res['total'],
                'correct': res['correct']}

    def inflect(self, sent):
        # output: token['oword']
        res = self.inf_decoder.decode(sent)


    ##################################################
    # contraction functions

    def train_contraction_one_step(self, sent):
        t0 = time()
        dy.renew_cg()
        self.encode_sent(sent, True)
        res = self.con_decoder.decode(sent, True)
        loss = res['loss']
        if loss:
            loss_value = loss.value()
            loss.backward()
            try:
                self.trainer.update()
            except:
                self.log('training error, load check point')
                self.load_model()
        else:
            loss_value = 0
        return {'time': time()-t0,
                'loss': loss_value,
                'total': res['total'],
                'correct': res['correct']}

    def contract(self, sent):
        # output: token['oword']
        res = self.con_decoder.decode(sent)


    ##################################################
    # generation functions
    def generate(self, sent):
    # def generate(self, sent, domain_key='linearized_domain'):
        # output: token['generated_domain']
        #         sent['generated_tokens']
        for token in traverse_bottomup(sent.root):
            seq = token['gold_linearized_domain'] or token['linearized_domain'] or token['domain']
            res = self.h2d_generator.decode(token, seq)
            token['generated_domain'] = res['seq']

        sent['generated_tokens'] = self.flatten(sent.root, 'generated_domain')


    def train_generation_one_step(self, sent):
        total = correct = loss_value = 0
        t0 = time()
        errs = []
        dy.renew_cg()
        self.encode_sent(sent, train_mode=True)

        for token in traverse_bottomup(sent.root):
            res = self.h2d_generator.decode(token, token['gold_linearized_domain'], token['lost'], train_mode=True)
            errs += res['errs']
            correct += res['correct']
            total += 1

        if errs:
            loss = dy.esum(errs)
            loss_value = loss.value()
            loss.backward()
            try:
                self.trainer.update()
            except:
                self.log('training error, load check point')
                self.load_model()


        return {'time': time()-t0,
                'loss': loss_value,
                'total': total,
                'correct': correct}

    ##################################################
    # linearization functions

    def linearize(self, sent):
        # output: token['linearized_domain']
        #         sent['linearized_tokens']
        # top-down traverse to sort each domain
        sent_agenda = [SentSequence(sent)]

        for token in traverse_topdown(sent.root):
            all_agendas = []
            ranks = {}

            if 'l2r' in self.args.lin_decoders:
                init_seq = self.l2r_linearizer.init_seq(token)
                agenda, _ = self.l2r_linearizer.decode(init_seq)
                all_agendas.append(agenda)

            if 'r2l' in self.args.lin_decoders:
                init_seq = self.r2l_linearizer.init_seq(token)
                agenda, _ = self.r2l_linearizer.decode(init_seq)
                all_agendas.append(agenda)

            if 'h2d' in self.args.lin_decoders:
                init_seq = self.h2d_linearizer.init_seq(token)
                agenda, _ = self.h2d_linearizer.decode(init_seq)
                all_agendas.append(agenda)

            best_seqs = self.vote_best_seq(sent, all_agendas, self.args.beam_size)
            token['linearized_domain'] = [t for t in best_seqs[0].linearized_tokens()] # remove <$$$>

            if self.args.ranker:
                new_agenda = []
                for sent_seq in sent_agenda:
                    for seq in best_seqs:
                        new_seq = sent_seq.append(seq)
                        new_agenda.append(new_seq)
                new_agenda.sort(key=lambda x: -x.score)
                sent_agenda = new_agenda[:self.args.beam_size]


        if self.args.ranker:

            random_seq = np.random.choice(sent_agenda)
            sent['random_linearized_tokens'] = random_seq.get_sorted_tokens()

            oracle_seq = min(sent_agenda, key=lambda x: x.get_inv_num())
            sent['oracle_linearized_tokens'] = oracle_seq.get_sorted_tokens()

            # greedy
            best_seq = max(sent_agenda, key=lambda x: x.score)
            sent['linearized_tokens'] = best_seq.get_sorted_tokens() 

            # reranked
            res = self.ranker.decode(sent_agenda)
            sent['ranked_linearized_tokens'] = res['seq'].get_sorted_tokens()

        else:
            sent['linearized_tokens'] = self.flatten(sent.root) # exclude root

    def vote_best_seq(self, sent, all_agendas, top=1):
        # TODO vote for generated seq as well
        all_seqs = defaultdict(float)
        ids2seq = {}
        for agenda in all_agendas:
            min_score = min(seq.score for seq in agenda)
            for seq in agenda:
                ids = seq.ids()
                all_seqs[ids] += (seq.score - min_score)
                if ids not in ids2seq or seq.score > ids2seq[ids].score:
                    ids2seq[ids] = seq

        sorted_ids = sorted(ids2seq, key=lambda x: -all_seqs[ids])
        sorted_seqs = [ids2seq[ids] for ids in sorted_ids]
        return sorted_seqs[:top]



    def flatten(self, token, key='linearized_domain'):
        assert key in ['linearized_domain', 'generated_domain', 'gold_linearized_domain', 'gold_generated_domain']
        if token['tid'] is None:
            return [token] # generated tokens
        else:
            return sum([(self.flatten(tk, key) if (tk is not token) else ([tk] if token['tid'] != 0 else [])) \
                     for tk in token[key]], [])

    def reorder(self, sent, key='linearized_tokens'):
        assert key in ['linearized_tokens', 'generated_tokens']
        new_tokens = []
        for t in sent[key]:
            if t['tid'] is None:
                t['tid'] = len(sent.tokens+new_tokens)
                new_tokens.append(t)
        mapping = {t['tid']: i for i, t in enumerate(sent[key], 1)}
        for t in sent[key]:
            t['tid'] = mapping[t['tid']] # new id
            t['hid'] = mapping.get(t['hid'], 0) # new head




    def train_linearization_one_step(self, sent):
        domain_total = domain_correct = loss_value = 0
        t0 = time()
        errs = []
        dy.renew_cg()
        self.encode_sent(sent, train_mode=True)

        sent_agenda = [SentSequence(sent)]

        for token in traverse_topdown(sent.root):
            all_agendas = []
            # training left-to-right
            if 'l2r' in self.args.lin_decoders:
                gold_seq = self.l2r_linearizer.init_seq(token)
                while not self.l2r_linearizer.finished(gold_seq):
                    agenda, gold_seq = self.l2r_linearizer.decode(gold_seq, True)
                    all_agendas.append(agenda)

                    if gold_seq is not agenda[0]:
                        scores = [gold_seq.score_expr] + [seq.score_expr for seq in agenda if seq is not gold_seq]
                        errs.append(dy.hinge(dy.concatenate(scores), 0))
            # right-to-left 
            if 'r2l' in self.args.lin_decoders:
                gold_seq = self.r2l_linearizer.init_seq(token)
                while not self.r2l_linearizer.finished(gold_seq):
                    agenda, gold_seq = self.r2l_linearizer.decode(gold_seq, True)
                    all_agendas.append(agenda)
                    if gold_seq is not agenda[0]:
                        scores = [gold_seq.score_expr] + [seq.score_expr for seq in agenda if seq is not gold_seq]
                        errs.append(dy.hinge(dy.concatenate(scores), 0))
            # head-to-dep
            if 'h2d' in self.args.lin_decoders:                         
                gold_seq = self.h2d_linearizer.init_seq(token)
                agenda = [gold_seq]
                while not self.h2d_linearizer.finished(gold_seq):
                    agenda, gold_seq = self.h2d_linearizer.decode(gold_seq, True)
                    all_agendas.append(agenda)
                    # update only against all incorrect sequences (exclude lower scoring gold seq)
                    if gold_seq is not agenda[0]:
                        scores = [gold_seq.score_expr] + [seq.score_expr for seq in agenda if not seq.correct]
                        errs.append(dy.hinge(dy.concatenate(scores), 0))

            # train ranker 
            if self.args.ranker:
                new_agenda = []
                best_seqs = self.vote_best_seq(sent, all_agendas, self.args.beam_size)
                for sent_seq in sent_agenda:
                    for seq in best_seqs:
                        new_seq = sent_seq.append(seq)
                        new_agenda.append(new_seq)
                new_agenda.sort(key=lambda x: -x.score)
                sent_agenda = new_agenda[:self.args.beam_size]

            if token['deps']:
                domain_total += 1
                domain_correct += agenda[0].correct

        if self.args.ranker:
            res = self.ranker.decode(sent_agenda, True)
            errs += res['errs']

        if errs:
            loss = dy.average(errs) if self.args.avg_loss else dy.esum(errs)
            loss_value = loss.value()
            loss.backward()
            try:
                self.trainer.update()
            except:
                self.log('training error, load check point')
                self.load_model()


        return {'time': time()-t0,
                'loss': loss_value,
                'total': domain_total,
                'correct': domain_correct
                }


    ##################################################
    # Auxiliary stuff

    def extract_rules(self, sents, min_freq=1, min_certainty=0.99):
        freq = defaultdict(lambda: defaultdict(int))
        for sent in sents:
            for t in sent.get_tokens():
                freq[f"{t['clemma']}-{t['upos']}-({'|'.join(t['morph'])})"][t['word']] += 1

        for k, v in sorted(freq.items(), key=lambda x: -sum(x[1].values())):
            if sum(v.values()) > min_freq:
                w, c = max(v.items(), key=lambda x:x[1])
                if c > sum(v.values()) * min_certainty: 
                    self.inf_rules[k] = w

        print(f'total inflection rules: {len(self.inf_rules)}')
        with gzip.open(self.args.model_file+'.lex.gz', 'wb') as stream:
            pickle.dump(self.inf_rules, stream, -1)


    def load_rules(self):
        with gzip.open(self.args.model_file+'.lex.gz','rb') as stream:
            self.inf_rules = pickle.load(stream)

    def eval_rules(self, sents):
        total = covered = correct = equal = 0
        for sent in sents:
            for t in sent.get_tokens():
                if t['word'] != t['clemma']:
                    total += 1
                    pat = f"{t['clemma']}-{t['upos']}-({'|'.join(t['morph'])})"
                    if pat in self.inf_rules:
                        covered += 1
                        correct += (t['word'] == self.inf_rules[pat])
                        if t['word'] != self.inf_rules[pat]:
                            print(f"{t['word']}\t{self.inf_rules[pat]}\t{pat}")
                else:
                    equal += 1

        self.log(f'total_equal={equal}, total_diff={total}, covered={covered}, correct={correct}')
        self.log(f'p={100*correct/covered:.2f}')
        self.log(f'r={100*correct/total:.2f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("mode", choices=['train', 'pred', 'eval'])
    parser.add_argument("-m", "--model_file")
    parser.add_argument("-t", "--train_file")
    parser.add_argument("-d", "--dev_file")
    parser.add_argument("-i", "--input_file")
    parser.add_argument("-p", "--pred_file")
    parser.add_argument("-g", "--gold_file")
    parser.add_argument("--tasks", default='lin', help='combinations of: lin, inf, gen')
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--max_step", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=32)
    parser.add_argument("--features", default='lemma+upos+label+morph')
    parser.add_argument("--tree_vecs", default='head+deps', help='combinations of: self, head, deps')
    parser.add_argument("--lin_decoders", default='h2d+l2r+r2l', help='combinations of: l2r, r2l, h2d')
    parser.add_argument("--pointer_type", default='glimpse', choices=['simple', 'glimpse', 'self'])
    parser.add_argument("--tree_lstm", default='att', choices=['simple', 'att', 'selfatt'])
    parser.add_argument("--head_input", default='deps_vec', choices=['vec', 'deps_vec', 'deps_mem'])

    # experimental: 
    parser.add_argument("--auto_xpos", action='store_true')
    parser.add_argument("--ignore_lemma_case", action='store_true')
    parser.add_argument("--lemmatize", action='store_true')
    parser.add_argument("--avg_loss", action='store_true')
    parser.add_argument("--no_rules", action='store_true')
    parser.add_argument("--no_seq", action='store_true')
    parser.add_argument("--no_lin_constraint", action='store_true')
    parser.add_argument("--ranker", action='store_true')


    args = parser.parse_args()

    # train a model
    if args.mode == 'train':
        assert args.model_file is not None 
        assert args.train_file is not None
        assert args.dev_file is not None
        model = Realization(args)
        model.train()
    # use the model to predict and write to a output file
    elif args.mode == 'pred':
        assert args.model_file is not None 
        assert args.input_file is not None 
        assert args.pred_file is not None 
        model = Realization(args)
        skip_lost = ('inf' not in model.args.tasks and 'con' not in model.args.tasks)
        print('skip_lost:', skip_lost)
        test_sents = read_conllu(args.input_file, False, skip_lost) # lemma, no_word

        # deal with korean lemma
        if model.args.lemmatize:
            print('lemmatize')
            convert_lemma_morph(test_sents)

        print('number of test sents:', len(test_sents))

        model.predict(test_sents, False)
        if 'con' in model.args.tasks:
            for sent in test_sents:
                tokens = sent['gold_linearized_tokens'] or sent['linearized_tokens'] or sent.get_tokens()
                capitalize(tokens, args.ignore_lemma_case)
            write_txt(args.pred_file, test_sents)
        else:
            write_conllu(args.pred_file, test_sents, ud=False, use_morphstr=True)
    # use the model to predict but only to evaluate
    elif args.mode == 'eval':
        assert args.model_file is not None 
        assert args.input_file is not None 
        model = Realization(args)
        skip_lost = ('inf' not in model.args.tasks and 'con' not in model.args.tasks)
        test_sents = read_conllu(args.input_file, False, skip_lost) # lemma, no_word

        # deal with korean lemma
        if model.args.lemmatize:
            print('lemmatize')
            convert_lemma_morph(test_sents)

        print('number of test sents:', len(test_sents))
        res = model.predict(test_sents)
        print(res)
    else:
        print('wrong mode!')

