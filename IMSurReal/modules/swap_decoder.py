import dynet as dy
import dynet_modules as dm
import numpy as np
from utils import *
from time import time
from collections import defaultdict

from modules.seq_encoder import SeqEncoder
from modules.bag_encoder import BagEncoder
from modules.tree_encoder import TreeEncoder


class SwapDecoder(Decoder):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.train_input_key = 'gold_projective_tokens' # or 'linearized tokens'
        self.train_output_key = 'gold_linearized_tokens'
        if 'gen' in self.args.tasks:
            self.pred_input_key = 'generated_tokens'
        if any(task in ['tsp', 'tsp-full', 'lin'] for task in self.args.tasks):
            self.pred_input_key = 'linearized_tokens'
        else:
            self.pred_input_key = 'gold_projective_tokens'
        self.pred_output_key = 'sorted_tokens'
        # self.vec_key = 'sum_tree'
        self.vec_key = 'swap_vec'

        if 'seq' in self.args.tree_vecs:
            self.seq_encoder = SeqEncoder(self.args, self.model, 'swap_seq')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder = BagEncoder(self.args, self.model, 'swap_bag')
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder = TreeEncoder(self.args, self.model, 'swap_tree')

        self.special = self.model.add_lookup_parameters((2, self.args.token_dim))
        self.f_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, model)
        self.b_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, model)
        self.mlp = dm.MLP(self.model, self.args.token_dim*2, 2, self.args.hid_dim)
        self.log(f'Initialized <{self.__class__.__name__}>, params = {self.model.parameter_count()}')

        self.stats = defaultdict(int)


    def decode(self, tokens, train_mode=False):
        # tmp id for each token to prevent repeated swap
        for i, t in enumerate(tokens, 1):
            t['tmp_id'] = i

        total = correct = 0
        errs = []

        st = Token({'tmp_id': 0, 'head': None})
        st.vecs['f_state'] = self.f_lstm.initial_state().add_input(self.special[0])
        bt = Token({'tmp_id': len(tokens)+1, 'head': None})
        bt.vecs[self.vec_key] = self.special[1]

        stk = [st]
        bfr = tokens + [bt]

        b_state = self.b_lstm.initial_state()

        for t in reversed(bfr):
            t.vecs['b_state'] = b_state = b_state.add_input(t.vecs[self.vec_key])

        while len(bfr) > 1:
            logit = self.mlp.forward(dy.concatenate([stk[-1].vecs['f_state'].output(), bfr[0].vecs['b_state'].output()]))
            pred = logit.npvalue().argmax()
            if train_mode:
                gold = self.tell(stk, bfr)
                total += 1
                correct += (pred == gold)
                errs.append(dy.pickneglogsoftmax(logit, gold))
                pred = gold
                self.stats['count'] += 1

            if len(stk) < 2 or stk[-1]['tmp_id'] > bfr[0]['tmp_id'] or pred == 0 or not allow_swap(stk, bfr): # shift
                stk, bfr = self.shift(stk, bfr)
                stk[-1].vecs['f_state'] = stk[0].vecs['f_state'].add_input(stk[-1].vecs[self.vec_key])
            else:
                stk, bfr = self.swap(stk, bfr)
                bfr[1].vecs['b_state'] = bfr[2].vecs['b_state'].add_input(bfr[1].vecs[self.vec_key])
                bfr[0].vecs['b_state'] = bfr[1].vecs['b_state'].add_input(bfr[0].vecs[self.vec_key])

        if train_mode:    
            self.stats['len'] += len(tokens)

        return {'seq': stk[1:],
                'loss': dy.esum(errs) if errs else None,
                'total': total,
                'correct': correct}

    def tell(self, stk, bfr):
        if len(stk) < 2 or stk[-1]['original_id'] < bfr[0]['original_id']:
            return 0 # shift
        else:
            return 1 # swap

    def shift(self, stk, bfr):
        return stk + bfr[:1], bfr[1:]

    def swap(self, stk, bfr):
        return stk[:-1], bfr[:1] + stk[-1:] + bfr[1:]


    def predict(self, sent, pipeline=False):
        # encode
        if 'seq' in self.args.tree_vecs:
            self.seq_encoder.encode(sent, 'linearized_tokens' if self.args.pred_seq else 'gold_linearized_tokens')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder.encode(sent)
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder.encode(sent, self.args.pred_tree)
        sum_vecs(sent, self.vec_key, ['feat', 'swap_seq', 'swap_bag', 'swap_tree'])

        tokens = sent[self.pred_input_key if pipeline else self.train_input_key]
        res = self.decode(tokens, False)
        sent[self.pred_output_key] = res['seq']


    def train_one_step(self, sent):
        t0 = time()

        # encode
        if 'seq' in self.args.tree_vecs:
            self.seq_encoder.encode(sent, 'linearized_tokens' if self.args.pred_seq else 'gold_linearized_tokens')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder.encode(sent)
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder.encode(sent, self.args.pred_tree)
        sum_vecs(sent, self.vec_key, ['feat', 'swap_seq', 'swap_bag', 'swap_tree'])

        # train on gold 
        tokens = sent[self.train_input_key]
        res = self.decode(tokens, True)

        return {'time': time()-t0,
                'loss': res['loss'].value() if res['loss'] else 0,
                'loss_expr': res['loss'],
                'total': res['total'],
                'correct': res['correct']
                }


    def evaluate(self, sents):
        gold_seqs = [sent[self.train_output_key] for sent in sents]
        input_seqs = [sent[self.pred_input_key] for sent in sents]
        pred_seqs = [sent[self.pred_output_key] for sent in sents]
        in_bleu = eval_all(gold_seqs, input_seqs)
        # bleu after sorting
        out_bleu = eval_all(gold_seqs, pred_seqs)
        self.log(f'input_bleu={100*in_bleu:.2f}, output_bleu={100*out_bleu:.2f}')
        self.log(f"len={self.stats['len']}, count={self.stats['count']}")
        self.stats = defaultdict(int)
        # return out_bleu - in_bleu
        return out_bleu


def allow_swap(stk, bfr):
    s, b = stk[-1], bfr[0]
    sh, bh = s['head'], b['head']
    if sh is None or bh is None:
        return False
    so, bo = sh['order'], bh['order']
    if sh is bh or sh is b:
        return s not in so or b not in so or so.index(s) > so.index(b) 
    elif bh is s:
        return b not in bo or s not in bo or bo.index(s) > bo.index(b)
    else:
        return True




