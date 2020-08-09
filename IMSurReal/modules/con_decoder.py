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

class ConDecoder(Decoder):
    def __init__(self, args, model, c2i, emb):
        super().__init__(args, model)

        self.train_input_key = 'gold_linearized_tokens'
        self.train_output_key = 'gold_contracted_tokens'
        self.pred_input_key = 'inflected_tokens' 
        self.pred_output_key = 'contracted_tokens'

        self.vec_key = 'inf_vec' 

        if 'seq' in self.args.tree_vecs:
            self.seq_encoder = SeqEncoder(self.args, self.model, 'con_seq')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder = BagEncoder(self.args, self.model, 'con_bag')
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder = TreeEncoder(self.args, self.model, 'con_tree')


        self.c2i = c2i
        self.i2c = list(self.c2i.keys())
        self.emb = emb
        self.dummies = model.add_lookup_parameters((4, self.args.token_dim))
        self.group_mlp = dm.MLP(self.model, 5*self.args.token_dim, 3, self.args.hid_dim)
        self.tok_lstm = dy.BiRNNBuilder(1, self.args.token_dim, self.args.token_dim, self.model, dy.VanillaLSTMBuilder)
        self.char_lstm = dy.BiRNNBuilder(1, self.args.char_dim, self.args.char_dim, self.model, dy.VanillaLSTMBuilder)
        self.init_c = self.model.add_parameters(2*self.args.char_dim)
        self.lstm_decoder = dy.VanillaLSTMBuilder(1, 2*self.args.char_dim+self.args.token_dim, self.args.char_dim, self.model)
        self.attention = dm.Attention(self.model, self.args.char_dim, self.args.char_dim, self.args.char_dim)
        self.contract_mlp = dm.MLP(self.model, self.args.char_dim, len(self.i2c), self.args.hid_dim)
        self.empty = self.model.add_parameters(self.args.char_dim)

    def encode(self, sent):
        # encode
        if 'seq' in self.args.tree_vecs:
            self.seq_encoder.encode(sent, 'linearized_tokens' if self.args.pred_seq else 'gold_linearized_tokens')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder.encode(sent)
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder.encode(sent, self.args.pred_tree)
        sum_vecs(sent, self.vec_key, ['feat', 'con_seq', 'con_bag', 'con_tree'])


    def decode(self, tokens, train_mode=False):
        total = correct = 0
        errs = []

        group = []
        for x, tk in enumerate(tokens):
            total += 1
            p1 = tokens[x-1].vecs[self.vec_key] if x > 0 else self.dummies[0]
            n1 = tokens[x+1].vecs[self.vec_key] if x < len(tokens)-1 else self.dummies[1]
            n2 = tokens[x+2].vecs[self.vec_key] if x < len(tokens)-2 else self.dummies[2]
            g1 = group[-1].vecs[self.vec_key] if group else self.dummies[3]
            x = dy.concatenate([tk.vecs[self.vec_key], p1, n1, n2, g1])
            logit = self.group_mlp.forward(x)
            pidx = logit.npvalue().argmax() 

            if train_mode:
                correct_conctract = 1
                # B (2) I (1) O (0)
                if tk['word'] == tk['cword']:
                    y = 0 # O
                    if group:
                        res = self.contract(group, True)
                        correct_conctract *= res['correct']
                        errs.append(res['loss'])
                        group = []
                elif group:
                    y = 1 # I
                    group.append(tk)
                else:
                    y = 2 # B
                    if group:
                        res = self.contract(group, True)
                        correct_conctract *= res['correct']
                        errs.append(res['loss'])
                        group = []
                    group.append(tk)
                err = dy.pickneglogsoftmax(logit, y)
                errs.append(err)
                correct += int(pidx == y) * correct_conctract
            else:
                if pidx == 0:
                    if group:
                        res = self.contract(group)
                        # change oword
                        group[0]['oword'] = res['out']
                        for t in group[1:]:
                            t['oword'] = ''
                        group = []
                elif pidx == 1:
                    group.append(tk)
                else:
                    if group:
                        res = self.contract(group)
                        # change oword
                        group[0]['oword'] = res['out']
                        for t in group[1:]:
                            t['oword'] = ''
                        group = []
                    group.append(tk)
        # correct prediction
        if not train_mode:
            for tk in tokens:
                correct += int(tk['oword'] == tk['cword'])

        return {'loss': dy.esum(errs) if errs else None,
                'correct': correct,
                'total': total}


    def contract(self, group, train_mode=False):
        correct = 1
        errs = []
        source = ' '.join([tk['word'] for tk in group])

        if train_mode:
            target = group[0]['cword']

        tok_vec = self.tok_lstm.transduce([tk.vecs[self.vec_key] for tk in group])[-1]

        char_vecs = [self.emb[self.c2i.get(c, 0)] for c in source]
        char_mat = dy.concatenate_cols(self.char_lstm.transduce(char_vecs))

        init_vec = dy.concatenate([tok_vec, self.init_c])
        s = self.lstm_decoder.initial_state().add_input(init_vec)
        dec_vec = self.empty
        out = ''

        for x in range(len(source)*2):
            enc_vec = self.attention.encode(char_mat, s.output())
            s = s.add_input(dy.concatenate([tok_vec, enc_vec, dec_vec]))
            logit = self.contract_mlp.forward(s.output())
            pidx = logit.npvalue().argmax()

            if train_mode:
                y = self.c2i[target[x]] if x < len(target) else 0
                err = dy.pickneglogsoftmax(logit, y)
                errs.append(err)
                correct *= int(pidx == y)
            else:
                y = pidx

            if y == 0:
                break
            else:
                out += self.i2c[y]
                dec_vec = self.emb[y]

        correct *= int(out == group[0]['cword']) # assuming it exists in dev data
        return {'out': out,
                'loss': dy.esum(errs) if errs else None,
                'correct': correct}



    def predict(self, sent, pipeline=False):
        # output: token['oword']
        if pipeline:
            tokens = sent[self.pred_input_key] or sent[self.train_input_key]
        else:
            tokens = sent[self.train_input_key] or sent['input_tokens']
        self.encode(sent)
        res = self.decode(tokens)
        sent[self.pred_output_key] = [t for t in tokens if t['oword'] and t.not_empty()]


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


    def evaluate(self, sents):
        gold_txts, pred_txts = [], []
        for sent in sents:
            # tokens = sent['gold_linearized_tokens'] or sent['gold_generated_tokens'] or sent.get_tokens()
            # gold_txts.append(' '.join([tk['cword'] for tk in tokens if tk['cword'] and tk.not_empty()]))
            # pred_txts.append(' '.join([tk['oword'] for tk in tokens if tk['oword'] and tk.not_empty()]))
            gold_txts.append(' '.join([t['cword'] for t in sent['gold_contracted_tokens']]))
            pred_txts.append(' '.join([t['oword'] for t in sent['contracted_tokens']]))
        return text_bleu(gold_txts, pred_txts)


