import dynet as dy
import dynet_modules as dm
import numpy as np
import gzip
from time import time
from utils import *
from collections import defaultdict
from modules.seq_encoder import SeqEncoder
from modules.bag_encoder import BagEncoder
from modules.tree_encoder import TreeEncoder

class InfDecoder(Decoder):
    def __init__(self, args, model, c2i, emb, inf_rules, dev_sents):
        super().__init__(args, model)
        self.pred_key = 'token-oword' # identify the prediction

        self.train_input_key = 'gold_linearized_tokens' # or 'linearized tokens'
        self.train_output_key = 'gold_linearized_tokens' # or 'linearized tokens'
        if 'swap' in self.args.tasks:
            self.pred_input_key = 'sorted_tokens'
        elif 'gen' in self.args.tasks:
            self.pred_input_key = 'generated_tokens'
        elif any(task in ['tsp', 'tsp-full', 'lin'] for task in self.args.tasks):
            self.pred_input_key = 'linearized_tokens'
        else:
            self.pred_input_key = 'input_tokens'
        self.pred_output_key = 'inflected_tokens'

        self.vec_key = 'inf_vec' 

        if 'seq' in self.args.tree_vecs:
            self.seq_encoder = SeqEncoder(self.args, self.model, 'inf_seq')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder = BagEncoder(self.args, self.model, 'inf_bag')
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder = TreeEncoder(self.args, self.model, 'inf_tree')

        self.c2i = c2i
        self.emb = emb
        self.i2e = ['<=>', '</$>', '✓', '✗'] + [c for c in c2i.keys()]
#        assert len(self.i2e) == len(set(self.i2e))
        self.e2i = {e: i for i, e in enumerate(self.i2e)}
        self.max_len = 100

        self.lstm_encoder = dy.BiRNNBuilder(1, self.args.char_dim, self.args.char_dim, self.model, dy.VanillaLSTMBuilder)
        self.lstm_decoder = dy.VanillaLSTMBuilder(1, 2*self.args.char_dim+self.args.token_dim, self.args.char_dim, self.model)
        self.attention = dm.Attention(self.model, self.args.char_dim+1, self.args.char_dim, self.args.char_dim)
        self.mlp = dm.MLP(self.model, self.args.char_dim, len(self.i2e), self.args.hid_dim)
        self.init_c = self.model.add_parameters(2*self.args.char_dim)
        self.empty = self.model.add_parameters(self.args.char_dim)

        self.log(f'Initialized <{self.__class__.__name__}>, params = {self.model.parameter_count()}')
        self.inf_rules = {}
        if not self.args.no_inf_rules:
            self.inf_rules = inf_rules 
            if dev_sents:
                self.eval_rules(dev_sents)

    def encode(self, sent):
        # encode
        if 'seq' in self.args.tree_vecs:
            self.seq_encoder.encode(sent, 'linearized_tokens' if self.args.pred_seq else 'gold_linearized_tokens')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder.encode(sent)
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder.encode(sent, self.args.pred_tree)
        sum_vecs(sent, self.vec_key, ['feat', 'inf_seq', 'inf_bag', 'inf_tree'])
        # sum_vecs(sent, self.vec_key, ['inf_seq', 'inf_bag', 'inf_tree'])


    def decode(self, tokens, train_mode=False):
        errs = []
        total = correct = 0

        for x, t in enumerate(tokens):
            word = self.inf_rules.get(f"{t['clemma']}-{t['upos']}-({'|'.join(t['morph'])})", '')
            use_rule = word != ''

            # train or predict without rule
            if train_mode or not word:
                word = ''
                cur = 1
                vecs = [self.emb[self.c2i.get(c, 0)] for c in t['clemma']]
                vecs = self.lstm_encoder.transduce(vecs)
                char_mat = dy.concatenate_cols(vecs)
                mask = np.zeros((1, len(vecs)))
                mask[0, 0] = 1

                init_vec = dy.concatenate([t.vecs[self.vec_key], self.init_c])
                s = self.lstm_decoder.initial_state().add_input(init_vec)
                dec_vec = self.empty
                j = 0

                for i in range(self.max_len):
                    input_mat = dy.concatenate([char_mat, dy.inputTensor(mask)])
                    enc_vec = self.attention.encode(input_mat, s.output())
                    s = s.add_input(dy.concatenate([t.vecs[self.vec_key], enc_vec, dec_vec]))
                    logit = self.mlp.forward(s.output())
                    pidx = logit.npvalue().argmax()

                    if train_mode:
                        y = 0 if (t['word'] == t['clemma']) else \
                           (1 if i >= len(t['diff']) else self.e2i.get(t['diff'][i], 4)) # unknown character
                        # avoid instability 
                        err = dy.pickneglogsoftmax(logit, y)
                        errs.append(err)
                        cur *= int(pidx == y)
                    else:
                        y = pidx

                    if y == 0: # identical word
                        word = t['clemma']
                        break
                    elif y == 1: # finish
                        break
                    elif y == 2: # copy
                        # fail safe for prediction
                        if j >= len(t['clemma']):
                            # word = t['clemma']
                            break
                        c = t['clemma'][j]
                        word += c
                        dec_vec = self.emb[self.c2i.get(c, 0)]
                        if j < len(t['clemma']):
                            mask[0, j] = 0
                        if j+1 < len(t['clemma']):
                            mask[0, j+1] = 1
                        j += 1
                    elif y == 3: # delete
                        # fail safe for prediction
                        if j >= len(t['clemma']):
                            # word = t['clemma']
                            break
                        dec_vec = self.empty
                        if j < len(t['clemma']):
                            mask[0, j] = 0
                        if j+1 < len(t['clemma']):
                            mask[0, j+1] = 1
                        j += 1
                    else: # add character
                        c = self.i2e[y]
                        word += c
                        dec_vec = self.emb[self.c2i.get(c, 0)]
                total += 1
                correct += cur if train_mode else (word == t['word'])
            else:
                if not train_mode:
                    total += 1
                    correct += (word == t['word'])
            t['oword'] = word

        return {'loss': dy.esum(errs) if errs else None,
                'correct': correct,
                'total': total}



    def train_one_step(self, sent):
        t0 = time()
        tokens = sent[self.train_input_key]
        self.encode(sent)
        res = self.decode(tokens, True)
        loss = res['loss']
        loss_value = loss.value() if loss else 0

        return {'time': time()-t0,
                'loss_expr': loss,
                'loss': loss_value,
                'total': res['total'],
                'correct': res['correct']}

    def predict(self, sent, pipeline=False):
        # output: token['oword']
        tokens = sent[self.pred_input_key if pipeline else self.train_input_key]
        self.encode(sent)
        res = self.decode(tokens)
        sent[self.pred_output_key] = sent[self.pred_input_key] # some harmless redundancy

    def evaluate(self, sents):
        # tokens = sents[0]['gold_linearized_tokens'] or sents[0]['gold_generated_tokens'] or sents[0].get_tokens()
        # print(' '.join(t['word'] for t in tokens))
        # print(' '.join(t['oword'] for t in tokens))
        tokens = sum([sent[self.pred_output_key] for sent in sents], [])
        total = len(tokens)
        correct = 0
        errs = []
        for tk in tokens:
            if tk['word'] == tk['oword']:
                correct += 1
            else:
                errs.append(tk)
        for tk in errs[:10]:
            print(f"{tk['word']}\t{tk['oword']}\t{tk['lemma']}\t{tk['morphstr']}")

        return correct / total


    # def evaluate(self, sents):
    #     # use bleu instead of acc for compatibility to other components in the pipeline
    #     gold_seqs = [sent[self.train_output_key] for sent in sents]
    #     pred_seqs = [sent[self.train_output_key] for sent in sents]
    #     # pred_seqs = [sent[self.pred_output_key if pipeline else self.train_output_key] for sent in sents]
        
    #     # accuracy 
    #     # total = correct = 0
    #     # for gs in gold_seqs:
    #     #     total += len(gs)
    #     #     correct += sum(int(t['word'] == t['oword'])for t in gs)
    #     # return correct / total
    #     return eval_all(gold_seqs, pred_seqs, 'word', 'oword')


    # def extract_rules(self, sents, min_freq=1, min_certainty=0.99):
    #     freq = defaultdict(lambda: defaultdict(int))
    #     for sent in sents:
    #         for t in sent[self.train_input_key]:
    #             freq[f"{t['clemma']}-{t['upos']}-({'|'.join(t['morph'])})"][t['word']] += 1

    #     for k, v in sorted(freq.items(), key=lambda x: -sum(x[1].values())):
    #         if sum(v.values()) > min_freq:
    #             w, c = max(v.items(), key=lambda x:x[1])
    #             if c > sum(v.values()) * min_certainty: 
    #                 self.inf_rules[k] = w

    #     self.log(f'total inflection rules: {len(self.inf_rules)}')
    #     with gzip.open(self.args.model_file+'.lex.gz', 'wb') as stream:
    #         pickle.dump(self.inf_rules, stream, -1)


    # def load_rules(self):
    #     with gzip.open(self.args.model_file+'.lex.gz','rb') as stream:
    #         self.inf_rules = pickle.load(stream)

    def eval_rules(self, sents):
        total = covered = correct = equal = 0
        for sent in sents:
            for t in sent[self.train_input_key]:
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
