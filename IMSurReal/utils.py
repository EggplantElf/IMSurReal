import sys
import dynet_modules as dm
from data import Token
import dynet as dy
import numpy as np
import gzip, pickle
import re
from collections import Counter
import nltk.translate.bleu_score as bs
import Levenshtein
from collections import defaultdict
from copy import copy

class ContractDecoder:
    def __init__(self, model, args, c2i, emb):
        self.model = model
        self.args = args
        self.c2i = c2i
        self.i2c = list(self.c2i.keys())
        self.emb = emb
        self.dummies = model.add_lookup_parameters((4, self.args.token_dim))
        self.group_mlp = dm.MLP(self.model, 5*self.args.token_dim, self.args.hid_dim, 3)
        self.tok_lstm = dy.BiRNNBuilder(1, self.args.token_dim, self.args.token_dim, self.model, dy.VanillaLSTMBuilder)
        self.char_lstm = dy.BiRNNBuilder(1, self.args.char_dim, self.args.char_dim, self.model, dy.VanillaLSTMBuilder)
        self.init_c = self.model.add_parameters(2*self.args.char_dim)
        self.lstm_decoder = dy.VanillaLSTMBuilder(1, 2*self.args.char_dim+self.args.token_dim, self.args.char_dim, self.model)
        self.attention = dm.Attention(self.model, self.args.char_dim, self.args.char_dim, self.args.char_dim)
        self.contract_mlp = dm.MLP(self.model, self.args.char_dim, self.args.hid_dim, len(self.i2c))
        self.empty = self.model.add_parameters(self.args.char_dim)

    def decode(self, sent, train_mode=False):
        total = correct = 0
        errs = []
        tokens = sent['gold_linearized_tokens'] or sent['linearized_tokens'] or sent.get_tokens()

        group = []
        for x, tk in enumerate(tokens):
            total += 1
            p1 = tokens[x-1]['vec'] if x > 0 else self.dummies[0]
            n1 = tokens[x+1]['vec'] if x < len(tokens)-1 else self.dummies[1]
            n2 = tokens[x+2]['vec'] if x < len(tokens)-2 else self.dummies[2]
            g1 = group[-1]['vec'] if group else self.dummies[3]
            x = dy.concatenate([tk['vec'], p1, n1, n2, g1])
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

        return {'loss': dy.average(errs) if errs else None,
                'correct': correct,
                'total': total}


    def contract(self, group, train_mode=False):
        correct = 1
        errs = []
        source = ' '.join([tk['word'] for tk in group])

        if train_mode:
            target = group[0]['cword']

        tok_vec = self.tok_lstm.transduce([tk['vec'] for tk in group])[-1]

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


class EditSeq2SeqDecoder:
    def __init__(self, rules, model, args, c2i, emb):
        self.rules = rules
        self.model = model
        self.args = args
        self.c2i = c2i
        self.emb = emb
        self.i2e = ['<=>', '</$>', '✓', '✗'] + [c for c in c2i.keys()]
        assert len(self.i2e) == len(set(self.i2e))
        self.e2i = {e: i for i, e in enumerate(self.i2e)}
        self.max_len = 100

        self.lstm_encoder = dy.BiRNNBuilder(1, self.args.char_dim, self.args.char_dim, self.model, dy.VanillaLSTMBuilder)
        # self.lstm_decoder = dy.VanillaLSTMBuilder(1, self.args.char_dim, self.args.char_dim, self.model)
        self.lstm_decoder = dy.VanillaLSTMBuilder(1, 2*self.args.char_dim+self.args.token_dim, self.args.char_dim, self.model)
        self.attention = dm.Attention(self.model, self.args.char_dim+1, self.args.char_dim, self.args.char_dim)
        self.mlp = dm.MLP(self.model, self.args.char_dim, self.args.hid_dim, len(self.i2e))
        self.init_w = self.model.add_parameters((self.args.char_dim, self.args.token_dim))
        self.init_b = self.model.add_parameters(self.args.char_dim)
        self.init_c = self.model.add_parameters(2*self.args.char_dim)
        self.empty = self.model.add_parameters(self.args.char_dim)
        # self.merge_w = self.model.add_parameters((self.args.char_dim, 2*self.args.char_dim+self.args.token_dim),
                    # init=dm.orthonormal_initializer(self.args.char_dim, 2*self.args.char_dim+self.args.token_dim))

        self.prev_dummy = self.model.add_parameters(self.args.token_dim)
        self.next_dummy = self.model.add_parameters(self.args.token_dim)
        self.merge_w = self.model.add_parameters((self.args.token_dim, 3*self.args.token_dim))


    def decode(self, sent, train_mode=False):
        errs = []
        total = correct = 0
        tokens = sent['gold_linearized_tokens'] or sent['linearized_tokens'] or sent.get_tokens()

        for x, t in enumerate(tokens):
            word = self.rules.get(f"{t['clemma']}-{t['upos']}-({'|'.join(t['morph'])})", '')
            use_rule = word != ''

            # train or predict without rule
            if train_mode or not word:
                word = ''
                cur = 1
                # vecs = [self.emb[self.c2i.get(c, 0)] for c in [t['upos']] + t['morph'] + list(t['lemma'])]
                vecs = [self.emb[self.c2i.get(c, 0)] for c in t['clemma']]
                vecs = self.lstm_encoder.transduce(vecs)
                char_mat = dy.concatenate_cols(vecs)
                mask = np.zeros((1, len(vecs)))
                mask[0, 0] = 1

                init_vec = dy.concatenate([t['vec'], self.init_c])
                s = self.lstm_decoder.initial_state().add_input(init_vec)
                dec_vec = self.empty
                j = 0

                for i in range(self.max_len):
                    input_mat = dy.concatenate([char_mat, dy.inputTensor(mask)])
                    enc_vec = self.attention.encode(input_mat, s.output())
                    s = s.add_input(dy.concatenate([t['vec'], enc_vec, dec_vec]))
                    logit = self.mlp.forward(s.output())
                    pidx = logit.npvalue().argmax()

                    if train_mode:
                        y = 0 if (t['word'] == t['clemma']) else \
                           (1 if i >= len(t['diff']) else self.e2i[t['diff'][i]])
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
            # if not use_rule and t['word'] != t['oword']:
            #     print(t['word'], t['oword'], t['upos'], t['morph'])

        # return {'loss': (dy.average(errs) if self.args.avg_loss else dy.esum(errs)) if errs else None,
        return {'loss': dy.average(errs) if errs else None,
                'correct': correct,
                'total': total}


class L2RLinearizer:
    def __init__(self, model, args):
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
                cand_mat = dy.concatenate_cols([t['vec'] for t in seq.rest])
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
    def __init__(self, model, args):
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
                cand_mat = dy.concatenate_cols([t['vec'] for t in seq.rest])
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
    def __init__(self, model, args):
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
        lstate = self.h2l_lstm.initial_state().add_input(token['vec'])
        rstate = self.h2r_lstm.initial_state().add_input(token['vec'])
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
                cand_mat = dy.concatenate_cols([t['vec'] for t in seq.rest])

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

            if train_mode and gold_seq is None:
                print('domain')
                print(seq.head['domain'], seq.head['gold_linearized_domain'], seq.head['r_order'])
                for tk in seq.head['r_order']:
                    print(list(tk.items()))
                print('prev agenda')
                for seq in agenda:
                    print(seq, seq.correct, seq.r_order)
                print('new agenda')
                for seq in new_agenda:
                    print(seq, seq.correct)
                exit()


            agenda = new_agenda[:self.args.beam_size]
            if train_mode and gold_seq not in agenda:
                break
        return agenda, gold_seq

class LostToken:
    # for generating new tokens
    def __init__(self, signature):
        self.lemma, self.upos, self.morphstr = signature
        self.morph = [] if self.morphstr == '_' else self.morphstr.split('|')
        # self.lemma = lemma

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

class SentRanker:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.global_lstm = dy.BiRNNBuilder(1, self.args.token_dim, self.args.token_dim, self.model, dy.VanillaLSTMBuilder)
        self.global_mlp = dm.MLP(self.model, self.args.token_dim, self.args.hid_dim, 1)

    def decode(self, seqs, train_mode=False):
        errs = []
        correct = 0
        for seq in seqs:
            vecs = [t['vec'] for t in seq.get_sorted_tokens()]
            vecs = self.global_lstm.transduce(vecs)
            seq.score_expr += sum(self.global_mlp.forward(v) for v in vecs)
            seq.score += seq.score_expr.value()

        best_seq = max(seqs, key=lambda x: x.score_expr.value())

        if train_mode:
            sorted_seqs = sorted(seqs, key=lambda x: x.get_inv_num())
            # if self.args.pairwise:
            #     for i in range(self.args.beam_size):
            #         s1, s2 = np.random.choice(seqs, 2, True)
            #         g = min(s1, s2, key=lambda x: x.get_inv_num()) 
            #         p = s1 if g is s2 else s2
            #         errs.append(dy.rectify(p.score_expr - g.score_expr + (p.get_inv_num() - g.get_inv_num())))
            # else:
            scores = dy.concatenate([seq.score_expr for seq in sorted_seqs])
            errs.append(dy.hinge(scores, 0))
            correct = (best_seq is sorted_seqs[0])

        return {'errs': errs,
                'seq': best_seq,
                'correct': correct}


class H2DGenerator:
    # Pipeline after linearizing the existing tokens, add the additional ones
    def __init__(self, model, args, lost_map):
        print('<H2DGenerator>')
        self.args = args
        # do b_lstm first, then use f_lstm to update seq states incrementally
        self.lf_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, model)
        self.lb_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, model)
        self.rf_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, model)
        self.rb_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, model)
        self.l_gen_mlp = dm.MLP(model, self.args.token_dim*2, self.args.hid_dim, len(lost_map))
        self.r_gen_mlp = dm.MLP(model, self.args.token_dim*2, self.args.hid_dim, len(lost_map))
        self.l_attention = dm.Attention(model, self.args.token_dim, self.args.token_dim, self.args.token_dim)
        self.r_attention = dm.Attention(model, self.args.token_dim, self.args.token_dim, self.args.token_dim)

        # for T2 generating lost tokens (no beam search for now since it's tricky to make the lost stable)
        self.begin = model.add_parameters(self.args.token_dim)
        self.end = model.add_parameters(self.args.token_dim)
        self.lost_map = lost_map
        self.lost_emb = model.add_lookup_parameters((len(self.lost_map), self.args.token_dim))
        self.gen_tokens = [LostToken(l) for l in self.lost_map]


    def decode(self, head, seq, targets = [], train_mode = False):
        out_lseq = []
        out_rseq = []
        errs = []
        correct = 1

        hidx = seq.index(head)
        lseq = seq[:hidx+1][::-1] + [{'lemma': '<###>','vec': self.begin, 'original_id':-999}] # h, l1, l2, ..., begin
        rseq = seq[hidx:] + [{'lemma': '<###>','vec': self.end, 'original_id':999}] # h, r1, r2, ..., end

        lb_vecs = self.lb_lstm.initial_state().transduce([tk['vec'] for tk in reversed(lseq)])
        for tk, lb_vec in zip(reversed(lseq), lb_vecs):
            tk['lb_vec'] = lb_vec
        rb_vecs = self.rb_lstm.initial_state().transduce([tk['vec'] for tk in reversed(rseq)])
        for tk, rb_vec in zip(reversed(rseq), rb_vecs):
            tk['rb_vec'] = rb_vec


        # generate left 
        lf_state = self.lf_lstm.initial_state().add_input(rseq[1]['rb_vec'])
        i = 0
        fk = lseq[0]
        while i < len(lseq)-1: 
            num_gen = 0
            bk = lseq[i+1]
            lf_state = lf_state.add_input(fk['vec'])
            lf_vec = lf_state.output()
            x = dy.concatenate([lf_vec, bk['lb_vec']])

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
                    fk['vec'] = self.lost_emb[gidx]
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
                    fk['vec'] = self.lost_emb[pidx]
                else:
                    num_gen = 0
                    i+=1
                    fk = lseq[i]
                if fk['lemma'] != '<###>':
                    out_lseq.append(fk)

        # generate right 
        rf_state = self.rf_lstm.initial_state().add_input(lseq[1]['lb_vec'])
        i = 0
        fk = rseq[0]
        while i < len(rseq)-1: 
            num_gen = 0
            bk = rseq[i+1]
            rf_state = rf_state.add_input(fk['vec'])
            rf_vec = rf_state.output()
            x = dy.concatenate([rf_vec, bk['rb_vec']])

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
                    fk['vec'] = self.lost_emb[gidx]
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
                    fk['vec'] = self.lost_emb[pidx]
                else:
                    num_gen = 0
                    i+=1
                    fk = rseq[i]
                if fk['lemma'] != '<###>':
                    out_rseq.append(fk)

        out_seq = out_lseq[::-1] + [head] + out_rseq

        return {'errs': errs, 
                'seq': out_seq,
                'correct': correct}




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
        state = self.state.add_input(tk['vec'])
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
        state = self.state.add_input(tk['vec'])
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
        lstate = self.lstate.add_input(tk['vec'])
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
        rstate = self.rstate.add_input(tk['vec'])
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


def traverse_topdown(h):
    yield h
    for d in h['deps']:
        if d.not_empty():
            yield from traverse_topdown(d)

def traverse_bottomup(h):
    for d in h['deps']:
        if d.not_empty():
            yield from traverse_bottomup(d)
    yield h


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


def eval_all(gold_seqs, pred_seqs, gkey='lemma', pkey='lemma'):
    all_ref = [ [[(token[gkey] if token['label'] != '<LOST>'else f"({token[gkey]})") for token in seq]] for seq in gold_seqs]
    all_hyp = [ [(token[pkey] if token['label'] != '<LOST>' else f"({token[pkey]})") for token in seq] for seq in pred_seqs]
    # all_ref = [ [[token[gkey] for token in seq]] for seq in gold_seqs]
    # all_hyp = [ [token[pkey] for token in seq] for seq in pred_seqs]
    for i in range(1):
        print(' '.join(all_ref[i][0]))
        print(' '.join(all_hyp[i]))

    chencherry = bs.SmoothingFunction()
    bleu = bs.corpus_bleu(all_ref, all_hyp, smoothing_function=chencherry.method2)
    return bleu

def text_bleu(gold_txts, pred_txts):
    all_ref = [[txt.lower().split()] for txt in gold_txts]
    all_hyp = [txt.lower().split() for txt in pred_txts]
    chencherry = bs.SmoothingFunction()
    bleu = bs.corpus_bleu(all_ref, all_hyp, smoothing_function=chencherry.method2)
    return bleu

def convert_lemma_morph(sents):
    # convert korean 
    for sent in sents:
        for t in sent.get_tokens():
            t['clemma'] = t['lemma'] # used for the characters in inflection
            stem = t['lemma'].split('+')[0]
            if stem:
                t['lemma'] = stem # use as feature
            xmorph = t['xpos'].split('+')
            t['morph'] += xmorph

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




