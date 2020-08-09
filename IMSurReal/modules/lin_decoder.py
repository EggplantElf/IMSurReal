import dynet as dy
import dynet_modules as dm
import numpy as np
import random
from utils import *
from time import time
from collections import defaultdict
from modules.seq_encoder import SeqEncoder
from modules.bag_encoder import BagEncoder
from modules.tree_encoder import TreeEncoder

class LinDecoder(Decoder):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.train_input_key = 'input_tokens'
        self.train_output_key = 'gold_linearized_tokens'
        self.pred_input_key = 'input_tokens'
        self.pred_output_key = 'linearized_tokens'
        if 'seq' in self.args.tree_vecs:
            self.seq_encoder = SeqEncoder(self.args, self.model, 'lin_seq')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder = BagEncoder(self.args, self.model, 'lin_bag')
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder = TreeEncoder(self.args, self.model, 'lin_tree')

        self.l2r_linearizer = L2RLinearizer(self.args, self.model) if 'l2r' in self.args.lin_decoders else None
        self.r2l_linearizer = R2LLinearizer(self.args, self.model) if 'r2l' in self.args.lin_decoders else None
        self.h2d_linearizer = H2DLinearizer(self.args, self.model) if 'h2d' in self.args.lin_decoders else None
        self.log(f'Initialized <{self.__class__.__name__}>, params = {self.model.parameter_count()}')

    def encode(self, sent):
        # encode
        if 'seq' in self.args.tree_vecs:
            self.seq_encoder.encode(sent, 'linearized_tokens' if self.args.pred_seq else 'gold_linearized_tokens')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder.encode(sent)
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder.encode(sent, self.args.pred_tree)
        sum_vecs(sent, 'lin_vec', ['feat', 'lin_seq', 'lin_bag', 'lin_tree'])


    def predict(self, sent, pipeline=False):
        # top-down traverse to sort each domain
        sent_agenda = [SentSequence(sent)]
        self.encode(sent)

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

            new_agenda = []
            for sent_seq in sent_agenda:
                for seq in best_seqs:
                    new_seq = sent_seq.append(seq)
                    new_agenda.append(new_seq)
            new_agenda.sort(key=lambda x: -x.score)
            sent_agenda = new_agenda[:self.args.beam_size]

        sent['nbest_linearized_tokens'] = [seq.get_sorted_tokens() for seq in sent_agenda]
        sent['linearized_tokens'] = sent['nbest_linearized_tokens'][0]

    def train_one_step(self, sent):
        domain_total = domain_correct = loss_value = 0
        t0 = time()
        errs = []

        self.encode(sent)
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

                if self.h2d_linearizer.finished(gold_seq):
                    all_agendas.append(agenda)
                else:
                    while not self.h2d_linearizer.finished(gold_seq):
                        agenda, gold_seq = self.h2d_linearizer.decode(gold_seq, True)
                        all_agendas.append(agenda)
                        # update only against all incorrect sequences (exclude lower scoring gold seq)
                        if gold_seq is not agenda[0]:
                            scores = [gold_seq.score_expr] + [seq.score_expr for seq in agenda if not seq.correct]
                            errs.append(dy.hinge(dy.concatenate(scores), 0))

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

        sent['nbest_linearized_tokens'] = [seq.get_sorted_tokens() for seq in sent_agenda]
        # random sequence from the beam to give the downstream training set more realistic input
        sent['linearized_tokens'] = random.choice(sent['nbest_linearized_tokens'])

        loss = dy.esum(errs) if errs else 0
        loss_value = loss.value() if loss else 0

        return {'time': time()-t0,
                'loss': loss_value,
                'loss_expr': loss,
                'total': domain_total,
                'correct': domain_correct
                }


    def evaluate(self, sents):
        gold_seqs = [sent[self.train_output_key] for sent in sents]
        pred_seqs = [sent[self.pred_output_key] for sent in sents]
        pred_bleu = eval_all(gold_seqs, pred_seqs)

        if 'nbest_linearized_tokens' in sents[0]:
            rand_seqs = [random.choice(sent['nbest_linearized_tokens']) for sent in sents]
            orac_seqs = [max([(sent_bleu(gs, ps), ps) for ps in sent['nbest_linearized_tokens']], key=lambda x: x[0])[1] \
                        for gs, sent in zip(gold_seqs, sents)]

            rand_bleu = eval_all(gold_seqs, rand_seqs)
            orac_bleu = eval_all(gold_seqs, orac_seqs)

            self.log(f'<PRED>{pred_bleu*100:.2f}')
            self.log(f'<RAND>{rand_bleu*100:.2f}')
            self.log(f'<ORAC>{orac_bleu*100:.2f}')
        return pred_bleu



    def vote_best_seq(self, sent, all_agendas, top=1):
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





class L2RLinearizer:
    def __init__(self, args, model):
        print('<L2RLinearizer>')
        self.args = args
        Pointer = {'simple': dm.SimplePointer, 'glimpse':dm.GlimpsePointer, 'self':dm.SelfPointer}[self.args.pointer_type]
        self.pointer = Pointer(model, self.args.token_dim)
        self.seq_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, model)
        self.init_vec = model.add_parameters(self.args.token_dim)

    def init_seq(self, token):
        return SequenceL2R(self.seq_lstm.initial_state().add_input(self.init_vec), token, [], token['domain'])

    def finished(self, seq):
        return len(seq.rest) == 0

    def decode(self, gold_seq, train_mode=False):
        agenda = [gold_seq]
        steps = len(gold_seq.rest)
        for i in range(steps):
            new_agenda = []
            for seq in agenda:
                cand_mat = dy.concatenate_cols([t.vecs['lin_vec'] for t in seq.rest])
                scores = self.pointer.point(seq.state.output(), cand_mat)
                # scores = dy.log_softmax(scores)
                for t, s in zip(seq.rest, scores):
                    if self.args.no_lin_constraint or seq.check_order(t):
                        new_seq = seq.append(t, s)
                        new_agenda.append(new_seq)
                        if train_mode and new_seq.is_gold():
                            gold_seq = new_seq
            new_agenda.sort(key=lambda x: -x.score)
            agenda = new_agenda[:self.args.beam_size]
            if train_mode and gold_seq not in agenda:
                break
        return agenda, gold_seq


class R2LLinearizer:
    def __init__(self, args, model):
        print('<R2LLinearizer>')
        self.args = args
        Pointer = {'simple': dm.SimplePointer, 'glimpse':dm.GlimpsePointer, 'self':dm.SelfPointer}[self.args.pointer_type]
        self.pointer = Pointer(model, self.args.token_dim)
        self.seq_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, model)
        self.init_vec = model.add_parameters(self.args.token_dim)

    def init_seq(self, token):
        return SequenceR2L(self.seq_lstm.initial_state().add_input(self.init_vec), token, [], token['domain'])

    def finished(self, seq):
        return len(seq.rest) == 0

    def decode(self, gold_seq, train_mode=False):
        agenda = [gold_seq]
        steps = len(gold_seq.rest)
        for i in range(steps):
            new_agenda = []
            for seq in agenda:
                cand_mat = dy.concatenate_cols([t.vecs['lin_vec'] for t in seq.rest])
                scores = self.pointer.point(seq.state.output(), cand_mat)
                # scores = dy.log_softmax(scores)
                for t, s in zip(seq.rest, scores):
                    if self.args.no_lin_constraint or seq.check_order(t):
                        new_seq = seq.append(t, s)
                        # print(new_seq, 'g' if new_seq.is_gold() else 'w')
                        new_agenda.append(new_seq)
                        if train_mode and new_seq.is_gold():
                            gold_seq = new_seq
            new_agenda.sort(key=lambda x: -x.score)
            agenda = new_agenda[:self.args.beam_size]
            if train_mode and gold_seq not in agenda:
                break
        return agenda, gold_seq


class H2DLinearizer:
    def __init__(self, args, model):
        print('<H2DLinearizer>')
        self.args = args
        Pointer = {'simple': dm.SimplePointer, 'glimpse':dm.GlimpsePointer, 'self':dm.SelfPointer}[self.args.pointer_type]
        self.l_pointer = Pointer(model, self.args.token_dim, self.args.token_dim)
        self.r_pointer = Pointer(model, self.args.token_dim, self.args.token_dim)

        self.h2l_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, model)
        self.h2r_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, model)

    def finished(self, seq):
        return len(seq.rest) == 0

    def init_seq(self, token):
        lstate = self.h2l_lstm.initial_state().add_input(token.vecs['lin_vec'])
        rstate = self.h2r_lstm.initial_state().add_input(token.vecs['lin_vec'])
        return SequenceH2D(lstate, rstate, token, [t for t in token['deps'] if t.not_empty()])
        # return SequenceH2D(lstate, rstate, token, token['deps'])

    def decode(self, gold_seq, train_mode=False):
        agenda = [gold_seq]
        steps = len(gold_seq.rest)
        for i in range(steps):
            new_agenda = []
            gold_seq = None
            ids2seq = {}
            for seq in agenda:
                cand_mat = dy.concatenate_cols([t.vecs['lin_vec'] for t in seq.rest])

                l_scores = self.l_pointer.point(seq.lstate.output(), cand_mat)
                r_scores = self.r_pointer.point(seq.rstate.output(), cand_mat)

                for t, s in zip(seq.rest, l_scores):
                    if self.args.no_lin_constraint or t not in seq.l_order or t is seq.l_order[-1]:
                        new_seq = seq.append_left(t, s)
                        ids = new_seq.ids()
                        if ids not in ids2seq or new_seq.score > ids2seq[ids].score:
                            ids2seq[ids] = new_seq
                        if train_mode and new_seq.is_gold() and (not gold_seq or new_seq.score > gold_seq.score):
                            gold_seq = new_seq

                for t, s in zip(seq.rest, r_scores):
                    if self.args.no_lin_constraint or t not in seq.r_order or t is seq.r_order[0]:
                        new_seq = seq.append_right(t, s)
                        ids = new_seq.ids()
                        if ids not in ids2seq or new_seq.score > ids2seq[ids].score:
                            ids2seq[ids] = new_seq
                        if train_mode and new_seq.is_gold() and (not gold_seq or new_seq.score > gold_seq.score):
                            gold_seq = new_seq

            new_agenda = list(ids2seq.values())
            new_agenda.sort(key=lambda x: -x.score)

            agenda = new_agenda[:self.args.beam_size]
            if train_mode and gold_seq not in agenda:
                break
        return agenda, gold_seq




class SequenceL2R: 
    def __init__(self, state, head, tokens, rest, lost_rest=[], prev=None): 
        self.state = state
        self.head = head
        self.tokens = tokens
        self.rest = rest
        self.gold_lost_rest = lost_rest
        self.prev = prev
        if prev is None:
            self.score = 0
            self.score_expr = 0
            self.correct = True
            self.gold_lost_rest = self.head['lost']
            self.required_order = self.head['order'][:]
        else:
            self.score = prev.score
            self.score_expr = prev.score_expr
            self.required_order = prev.required_order[:]

    def __repr__(self):
        return ' '.join(str(t['original_id']) for t in self.tokens) + '(' +' '.join(str(t['original_id']) for t in self.rest) + ')'

    def ids(self):
        return tuple(t['tid'] for t in self.tokens)

    def lemmas(self):
        return tuple(t['lemma'] for t in self.tokens)

    def oids(self):
        return [t['original_id'] for t in self.tokens]

    def linearized_tokens(self):
        return self.tokens

    def check_order(self, tk):
        return tk not in self.required_order or tk is self.required_order[0]

    def append(self, tk, s):
        state = self.state.add_input(tk.vecs['lin_vec'])
        lost_rest = [t for t in self.gold_lost_rest if t['original_id'] != tk['original_id']] # non-empty only in training
        seq = SequenceL2R(state, self.head, self.tokens+[tk], [t for t in self.rest if t is not tk], lost_rest, self)
        seq.score_expr += s
        seq.score += s.value()
        if tk in seq.required_order:
            seq.required_order.remove(tk)
        return seq

    def is_gold(self, lost=False):
        rest_ids = [t['original_id'] for t in self.rest + (self.gold_lost_rest if lost else [])]
        self.correct = self.prev.correct and all(self.tokens[-1]['original_id'] < i for i in rest_ids)
        return self.correct


class SequenceR2L: 
    def __init__(self, state, head, tokens, rest, prev=None): 
        self.state = state
        self.head = head
        self.tokens = tokens
        self.rest = rest
        self.prev = prev
        if prev is None:
            self.score = 0
            self.score_expr = 0
            self.correct = True
            self.required_order = self.head['order'][:]
        else:
            self.score = prev.score
            self.score_expr = prev.score_expr
            self.required_order = prev.required_order[:]

    def __repr__(self):
        return ' '.join(str(t['original_id']) for t in self.tokens) + '(' +' '.join(str(t['original_id']) for t in self.rest) + ')'

    def linearized_tokens(self):
        return self.tokens

    def ids(self):
        return tuple(t['tid'] for t in self.tokens)

    def lemmas(self):
        return tuple(t['lemma'] for t in self.tokens)

    def oids(self):
        return [t['original_id'] for t in self.tokens]

    def check_order(self, tk):
        return tk not in self.required_order or tk is self.required_order[-1]

    def append(self, tk, s):
        state = self.state.add_input(tk.vecs['lin_vec'])
        seq = SequenceR2L(state, self.head, [tk]+self.tokens, [t for t in self.rest if t is not tk], self)
        seq.score_expr += s
        seq.score += s.value()
        if tk in seq.required_order:
            seq.required_order.remove(tk)
        return seq

    def is_gold(self):
        self.correct = self.prev.correct and all(self.tokens[0]['original_id'] > t['original_id'] for t in self.rest)
        return self.correct


class SequenceH2D: 
    """
    Double-ended Sequence, starts with the head token, 
    appends dependents on both sides from near to far,
    allows spurious ambiguity of the gold sequence,
    """

    def __init__(self, lstate, rstate, head, rest, lost_rest=[], ldeps=[], rdeps=[], prev=None): 
        self.lstate = lstate
        self.rstate = rstate
        self.head = head
        self.ldeps = ldeps # grow inside-out
        self.rdeps = rdeps # grow inside-out
        self.rest = rest
        self.gold_lost_rest = lost_rest
        self.prev = prev
        if prev is None:
            self.score = 0
            self.score_expr = 0
            self.correct = True
            self.l_order = self.head['l_order'][:]
            self.r_order = self.head['r_order'][:]
            self.gold_lost_rest = self.head['lost']
            # print('lost', [t['original_id'] for t in self.head['lost']])
        else:
            self.score = prev.score
            self.score_expr = prev.score_expr
            self.l_order = prev.l_order[:]
            self.r_order = prev.r_order[:]
            self.correct = prev.correct

    def ids(self):
        return tuple(t['tid'] for t in self.ldeps + [self.head] + self.rdeps)
        # return tuple(t['tid'] for t in self.linearized_tokens())

    def oids(self):
        return [t['original_id'] for t in self.ldeps + [self.head] + self.rdeps]

    def linearized_tokens(self):
        # all content tokens (excluding <$$$>)
        return [t for t in (self.ldeps + [self.head] + self.rdeps) if t['lemma'] != '<$$$>']

    def __repr__(self):
        return ' '.join(str(t) for t in self.ldeps) + \
               '<' + str(self.head) + '>' + \
               ' '.join(str(t) for t in self.rdeps) + \
               ' [' + ' '.join(str(t) for t in self.rest) + ']' +\
               ' {' + ' '.join(str(t) for t in self.gold_lost_rest) + '}'


    def lmost(self):
        return self.ldeps[0] if self.ldeps else self.head

    def rmost(self):
        return self.rdeps[-1] if self.rdeps else self.head

    def append_left(self, tk, s):
        lstate = self.lstate.add_input(tk.vecs['lin_vec'])
        rstate = self.rstate
        ldeps = [tk] + self.ldeps
        rdeps = self.rdeps
        lost_rest = [t for t in self.gold_lost_rest if t['original_id'] != tk['original_id']] # non-empty only in training
        seq = SequenceH2D(lstate, rstate, self.head, [t for t in self.rest if t is not tk], lost_rest, ldeps, rdeps, self)
        seq.score_expr += s
        seq.score += s.value()
        if tk in seq.l_order:
            seq.l_order.remove(tk)
        return seq

    def append_right(self, tk, s):
        lstate = self.lstate
        rstate = self.rstate.add_input(tk.vecs['lin_vec'])
        ldeps = self.ldeps
        rdeps = self.rdeps + [tk]
        lost_rest = [t for t in self.gold_lost_rest if t['original_id'] != tk['original_id']] # non-empty only in training
        seq = SequenceH2D(lstate, rstate, self.head, [t for t in self.rest if t is not tk], lost_rest, ldeps, rdeps, self)
        seq.score_expr += s
        seq.score += s.value()
        if tk in seq.r_order:
            seq.r_order.remove(tk)
        return seq

    def is_gold(self, lost=False):
        lmost, rmost = self.lmost(), self.rmost()
        rest_ids = [t['original_id'] for t in self.rest + (self.gold_lost_rest if lost else [])]

        ids = [t['original_id'] for t in self.linearized_tokens()]

        if lmost['lemma'] == '<$$$>' and rest_ids and min(rest_ids) < lmost['original_id']:
            self.correct = False
        elif rmost['lemma'] == '<$$$>' and rest_ids and max(rest_ids) > rmost['original_id']:
            self.correct = False
        else:
            self.correct = self.prev.correct and len(ids) == len(set(ids)) and ids == sorted(ids) and \
                            not any(min(ids) < tid < max(ids) for tid in rest_ids)
        return self.correct

class SentSequence:
    def __init__(self, sent, domain_seqs = {}):
        self.sorted_tokens = []
        self.sent = sent
        self.score_expr = 0
        self.score = 0
        self.inv_num = None
        self.domain_seqs = domain_seqs
        if not self.domain_seqs:
            self.domain_seqs[0] = self.sent.root['deps']

    def append(self, domain_seq):
        new_seq = SentSequence(self.sent, copy(self.domain_seqs))
        new_seq.domain_seqs[domain_seq.head['tid']] = domain_seq.linearized_tokens()
        new_seq.score_expr = self.score_expr + domain_seq.score_expr
        new_seq.score = self.score + domain_seq.score
        return new_seq

    def is_gold(self):
        return all(seq.correct for seq in self.domain_seqs)

    def get_sorted_tokens(self):
        if not self.sorted_tokens:
            self.sorted_tokens = self.flatten(self.sent.root)            
        return self.sorted_tokens

    def get_inv_num(self):
        if self.inv_num is None:
            self.inv_num = inverse_num(self.get_sorted_tokens()) ** 0.5
        return self.inv_num

    def flatten(self, head):
        return sum([(self.flatten(tk) if (tk is not head) else ([tk] if tk['tid'] else []) ) \
                     for tk in self.domain_seqs[head['tid']]], [])

