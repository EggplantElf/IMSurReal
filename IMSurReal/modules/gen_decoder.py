import dynet as dy
import dynet_modules as dm
import numpy as np
from time import time
from utils import *
from data import *
from collections import defaultdict
from modules.seq_encoder import SeqEncoder
from modules.bag_encoder import BagEncoder
from modules.tree_encoder import TreeEncoder

class GenDecoder(Decoder):
    def __init__(self, args, model, lost_map):
        super().__init__(args, model)
        self.lost_map = lost_map
        self.train_input_key = 'gold_linearized_domain'
        self.train_output_key = 'gold_generated_tokens'
        if any(task in ['tsp', 'tsp-full', 'lin'] for task in self.args.tasks):
            self.pred_input_key = 'linearized_domain'
        else:
            self.pred_input_key = 'domain'
        self.pred_output_key = 'generated_tokens'
        self.vec_key = 'gen_vec'

        if 'seq' in self.args.tree_vecs:
            self.seq_encoder = SeqEncoder(self.args, self.model, 'gen_seq')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder = BagEncoder(self.args, self.model, 'gen_bag')
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder = TreeEncoder(self.args, self.model, 'gen_tree')


        self.lf_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, self.model)
        self.lb_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, self.model)
        self.rf_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, self.model)
        self.rb_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, self.model)
        self.l_gen_mlp = dm.MLP(self.model, self.args.token_dim*2, len(lost_map), self.args.hid_dim)
        self.r_gen_mlp = dm.MLP(self.model, self.args.token_dim*2, len(lost_map), self.args.hid_dim)
        self.l_attention = dm.Attention(self.model, self.args.token_dim, self.args.token_dim, self.args.token_dim)
        self.r_attention = dm.Attention(self.model, self.args.token_dim, self.args.token_dim, self.args.token_dim)

        # for T2 generating lost tokens (no beam search for now since it's tricky to make the lost stable)
        self.special = self.model.add_lookup_parameters((2, self.args.token_dim))
        self.end = self.model.add_parameters(self.args.token_dim)
        self.lost_map = lost_map
        self.lost_emb = self.model.add_lookup_parameters((len(self.lost_map), self.args.token_dim))
        self.gen_tokens = [LostToken(l) for l in self.lost_map]


        self.log(f'Initialized <{self.__class__.__name__}>, params = {self.model.parameter_count()}')

    def encode(self, sent):
        # encode
        if 'seq' in self.args.tree_vecs:
            self.seq_encoder.encode(sent, 'linearized_tokens' if self.args.pred_seq else 'gold_linearized_tokens')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder.encode(sent)
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder.encode(sent, self.args.pred_tree)
        sum_vecs(sent, self.vec_key, ['feat', 'gen_seq', 'gen_bag', 'gen_tree'])


    def decode(self, head, seq, targets = [], train_mode = False):
        out_lseq = []
        out_rseq = []
        errs = []
        correct = 1

        hidx = seq.index(head)

        bt = Token({'lemma': '<###>', 'original_id':-999})
        bt.vecs[self.vec_key] = self.special[0]
        et = Token({'lemma': '<###>', 'original_id':999})
        et.vecs[self.vec_key] = self.special[1]
        lseq = seq[:hidx+1][::-1] + [bt] # h, l1, l2, ..., begin
        rseq = seq[hidx:] + [et] # h, r1, r2, ..., end

        # from left backwards (begin to head)
        lb_vecs = self.lb_lstm.initial_state().transduce([tk.vecs[self.vec_key] for tk in reversed(lseq)])
        for tk, lb_vec in zip(reversed(lseq), lb_vecs):
            tk.vecs['lb'] = lb_vec
        # from right backwards (end to head)
        rb_vecs = self.rb_lstm.initial_state().transduce([tk.vecs[self.vec_key] for tk in reversed(rseq)])
        for tk, rb_vec in zip(reversed(rseq), rb_vecs):
            tk.vecs['rb'] = rb_vec


        # generate left 
        lf_state = self.lf_lstm.initial_state().add_input(rseq[1].vecs['rb'])
        i = 0
        fk = lseq[0]
        num_gen = 0
        while i < len(lseq)-1: 
            bk = lseq[i+1]
            lf_state = lf_state.add_input(fk.vecs[self.vec_key])
            lf_vec = lf_state.output()
            x = dy.concatenate([lf_vec, bk.vecs['lb']])

            logit = self.l_gen_mlp.forward(x)

            pidx = logit.npvalue().argmax()
            if train_mode:
                # reverse for left
                gts = [t for t in targets if fk['original_id'] > t['original_id'] > bk['original_id']]
                # generate target
                if gts:
                    gt = max(gts, key=lambda x:x['original_id']) # right-most
                    gidx = self.lost_map[signature(gt)]
                    targets = [t for t in targets if t is not gt]
                    fk = gt # next left context
                    fk.vecs[self.vec_key] = self.lost_emb[gidx]
                # next token
                else:
                    gidx = 0 
                    i += 1
                    fk = lseq[i]

                err = dy.pickneglogsoftmax(logit, gidx)
                errs.append(err)
                correct *= (pidx == gidx)
            else:
                # prevent generating too many tokens
                if pidx > 0 and num_gen <= 5:
                    num_gen += 1
                    fk = self.gen_tokens[pidx].generate(head['tid']) # oid=None, tid=None
                    fk.vecs[self.vec_key] = self.lost_emb[pidx]
                else:
                    num_gen = 0
                    i+=1
                    fk = lseq[i]
                if fk['lemma'] != '<###>':
                    out_lseq.append(fk)

        # generate right 
        rf_state = self.rf_lstm.initial_state().add_input(lseq[1].vecs['lb'])
        i = 0
        fk = rseq[0]
        num_gen = 0
        while i < len(rseq)-1: 
            bk = rseq[i+1]
            rf_state = rf_state.add_input(fk.vecs[self.vec_key])
            rf_vec = rf_state.output()
            x = dy.concatenate([rf_vec, bk.vecs['rb']])

            logit = self.r_gen_mlp.forward(x)
            pidx = logit.npvalue().argmax()
            if train_mode:
                # reverse for left
                gts = [t for t in targets if fk['original_id'] < t['original_id'] < bk['original_id']]
                # generate target
                if gts:
                    gt = min(gts, key=lambda x:x['original_id']) # right-most
                    # gidx = self.lost_map[gt['lemma']] # why no error? 
                    gidx = self.lost_map[signature(gt)]
                    targets = [t for t in targets if t is not gt]
                    fk = gt # next left context
                    fk.vecs[self.vec_key] = self.lost_emb[gidx]
                # next token
                else:
                    gidx = 0 
                    i += 1
                    fk = rseq[i]

                err = dy.pickneglogsoftmax(logit, gidx)
                errs.append(err)
                correct *= (pidx == gidx)
            else:
                # prevent generating too many tokens
                if pidx > 0 and num_gen <= 5:
                    num_gen += 1
                    fk = self.gen_tokens[pidx].generate(head['tid']) # oid=None, tid=None
                    fk.vecs[self.vec_key] = self.lost_emb[pidx]
                else:
                    num_gen = 0
                    i+=1
                    fk = rseq[i]
                if fk['lemma'] != '<###>':
                    out_rseq.append(fk)

        out_seq = out_lseq[::-1] + [head] + out_rseq

        return {'loss': dy.esum(errs) if errs else None,
                'seq': out_seq,
                'correct': correct}

    def predict(self, sent, pipeline=False):
        # output: token['generated_domain']
        #         sent['generated_tokens']

        self.encode(sent)
        for token in traverse_bottomup(sent.root):
            # seq = token[self.pred_input_key if pipeline else self.train_input_key]
            seq = token[self.train_input_key] or token[self.pred_input_key]
            res = self.decode(token, seq)
            token['generated_domain'] = res['seq']

        sent[self.pred_output_key] = flatten(sent.root, 'generated_domain')


    def train_one_step(self, sent):
        total = correct = loss = 0
        t0 = time()

        self.encode(sent)
        for token in traverse_bottomup(sent.root):
            res = self.decode(token, token[self.train_input_key], token['lost'], train_mode=True)
            loss += res['loss']
            total += 1
            correct += res['correct']

        loss_value = loss.value() if loss else 0

        return {'time': time()-t0,
                'loss': loss_value,
                'loss_expr': loss,
                'total': total,
                'correct': correct
                }


    def evaluate(self, sents):
        gold_seqs = [sent[self.train_output_key] for sent in sents]
        pred_seqs = [sent[self.pred_output_key] for sent in sents]
        return eval_all(gold_seqs, pred_seqs)

