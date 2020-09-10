import sys
from data import *
from utils import *
import dynet as dy
import dynet_modules as dm
import numpy as np
from argparse import ArgumentParser
import gzip, pickle
from time import time 
import re
from collections import defaultdict
import Levenshtein
import random
import json
from modules.feat_encoder import FeatEncoder
from modules.tree_encoder import TreeEncoder
from modules.tsp_decoder import TSPDecoder
from modules.inf_decoder import InfDecoder
from modules.con_decoder import ConDecoder
from modules.swap_decoder import SwapDecoder
from modules.lin_decoder import LinDecoder
from modules.gen_decoder import GenDecoder
from tqdm import tqdm

class Realization(object):
    def __init__(self, args):
        self.args = args
        self.model = dy.Model()
        self.encoders = {}
        self.decoders = {}

        self.train_sents = None
        self.dev_sents = None
        if args.mode == 'train':
            self.args.tasks = re.split(r',|\+', self.args.tasks)
            self.args.features = re.split(r',|\+', self.args.features)
            self.args.tree_vecs = re.split(r',|\+', self.args.tree_vecs)

            skip_lost = ('inf' not in  self.args.tasks and 'con' not in self.args.tasks)

            # generator for simplified training data, only for extracting maps
            self.simple_train_sents = read_conllu(self.args.train_file, self.args.ud_train, skip_lost, self.args.orig_word, 
                                            self.args.lemmatize, True, self.args.first_train)
            # generator for real training data
            self.train_sents = read_conllu(self.args.train_file, self.args.ud_train, skip_lost, self.args.orig_word, 
                                            self.args.lemmatize, False, self.args.first_train)

            # self.extra_sents = read_conllu(self.args.extra_file, True, skip_lost, self.args.orig_word, self.args.first_extra) if self.args.extra_file else []\

            # To save memory, do not keep all the training senteces, but iterate through them
            # First, fast iterate through training sentences to extract the features
            # Later iterate again for training

            self.dev_sents = list(read_conllu(self.args.dev_file, self.args.ud_dev, skip_lost, self.args.orig_word, self.args.lemmatize))
            self.test_sents = list(read_conllu(self.args.input_file, self.args.ud_test, skip_lost, self.args.orig_word, self.args.lemmatize)) if self.args.input_file else []
            
            # initialize feat_encoder here, because we need look up features in the data
            # iterate through the training data once
            t0 = time()
            self.encoders['feat'] = FeatEncoder(self.args, self.model, self.simple_train_sents)
            self.log(f'Time used for creating map: {(time()-t0):.1f}s')

            self.log(f'train sents: {self.args.num_train_sents}')
            # self.log(f'extra sents: {len(self.extra_sents)}')
            self.log(f'dev sents: {len(self.dev_sents)}')
            self.log(f'test sents: {len(self.test_sents)}')

            # check oov 
            woov = loov = total = 0
            for sent in self.dev_sents:
                for t in sent.get_tokens():
                    total += 1
                    if t['word'] not in self.encoders['feat'].word_map:
                        woov += 1
                    if t['lemma'] not in self.encoders['feat'].lemma_map:
                        loov += 1
            self.log(f'Word OOV: {woov} / {total} = {100*woov/total:.2f}')
            self.log(f'Lemma OOV: {loov} / {total} = {100*loov/total:.2f}')

            # self.setup_pipeline()

            self.save_args()
        else:
            self.load_args()
            self.encoders['feat'] = FeatEncoder(self.args, self.model)


        # Decoders
        if 'lin' in self.args.tasks:
            self.decoders['lin'] = LinDecoder(self.args, self.model)
        if 'tsp' in self.args.tasks:
            self.decoders['tsp'] = TSPDecoder(self.args, self.model, full=False)
        if 'tsp-full' in self.args.tasks:
            self.decoders['tsp-full'] = TSPDecoder(self.args, self.model, full=True)
        if 'swap' in self.args.tasks:
            self.decoders['swap'] = SwapDecoder(self.args, self.model)
        if 'gen' in self.args.tasks:
            self.decoders['gen'] = GenDecoder(self.args, self.model, self.encoders['feat'].lost_map)
        if 'inf' in self.args.tasks:
            self.decoders['inf'] = InfDecoder(self.args, self.model, 
                                              self.encoders['feat'].char_map, self.encoders['feat'].char_emb,
                                              self.encoders['feat'].inf_rules, self.dev_sents)
        if 'con' in self.args.tasks:
            self.decoders['con'] = ConDecoder(self.args, self.model,
                                              self.encoders['feat'].char_map, self.encoders['feat'].char_emb)


        self.log(f'Total params: {self.model.parameter_count()}')

        self.trainer = dy.AdamTrainer(self.model)
        self.trainer.set_clip_threshold(1)
        print(self.args)

        if args.mode != 'train':
            self.load_model()

        print(f'tasks = {self.args.tasks}')
        print(f'train_file = {self.args.train_file}')
        # print(f'extra_file = {self.args.extra_file}')
        print(f'dev_file = {self.args.dev_file}')
        print(f'input_file = {self.args.input_file}')
        print(f'pred_file = {self.args.pred_file}')

    def save_args(self):
        with gzip.open(self.args.model_file+'.args.gz', 'wb') as stream:
            pickle.dump(self.args, stream, -1)

    def load_args(self):
        with gzip.open(self.args.model_file+'.args.gz','rb') as stream:
            keep_args = self.args
            self.args = pickle.load(stream)
            self.args.mode = keep_args.mode
            self.args.model_file = keep_args.model_file
            self.args.input_file = keep_args.input_file            
            self.args.gold_file = keep_args.gold_file            
            self.args.pred_file = keep_args.pred_file

    def save_model(self):
        self.log('saving model')
        self.model.save(self.args.model_file)
        self.show_l2_norms()


    def load_model(self):
        self.log('loading model')
        self.model.populate(self.args.model_file)
        self.show_l2_norms()

    def log(self, msg):
        if self.args.mode == 'train' and self.args.model_file:
            with open(self.args.model_file+'.log', 'a') as f:
                f.write(str(msg)+'\n')
        print(msg)

    def show_l2_norms(self):
        total = 0
        for k, m in self.encoders.items():
            norm = m.l2_norm()
            v = norm.value() / m.model.parameter_count() if norm else 0
            total += v
            print(f'<Encoder:{k}> = {v:.4f}')
        for k, m in self.decoders.items():
            v = m.l2_norm().value() / m.model.parameter_count() 
            total += v
            print(f'<Decoder:{k}> = {v:.4f}')
        print(f'<Total> = {total:.2f}')


    def encode_sent(self, sent, train_mode=False):
        # only encode the features for sharing, let each decoder encode its own structure
        dy.renew_cg()
        self.encoders['feat'].encode(sent, train_mode)

    def train(self):

        self.log(f'Before training')
        self.show_l2_norms()

        all_loss = defaultdict(float)
        all_correct = defaultdict(float)
        all_total = defaultdict(float)
        waited = 0
        best_score = -1
        step = 0

        for batch in iterate_batch(self.train_sents, self.args.eval_every):
            # train on a batch of sentences
            t0 = time()
            for sent in tqdm(batch):
                step += 1
                self.encode_sent(sent, True)
                loss = 0
                for task in self.args.tasks:
                    res = self.decoders[task].train_one_step(sent)
                    all_loss[task] += res['loss']
                    all_total[task] += res['total']
                    all_correct[task] += res['correct']
                    loss += res['loss_expr']
                    
                # # clear prediction after training to save memory                        
                # sent.clear_pred()

                if loss:
                    # if self.args.regularize:
                    #     # only regularize dense params of all encoders and decoders
                    #     l2norm = sum(dy.sum_elems(p**2) for p in self.model.parameters_list())
                    #     l2norm_value = self.args.regularize * l2norm.value()
                    #     loss += self.args.regularize * l2norm
                    # else:
                    #     l2norm_value = 0
                    try:
                        loss.backward()
                        self.trainer.update()
                    except:
                        self.log('bad gradient, load previous model')
                        self.load_model()
            train_time = time() - t0

            

            # evaluate on dev set
            t0 = time()
            res = self.predict_all(self.dev_sents[:1000], pipeline=True)
            test_time = time() - t0
            self.log(f"[step={step}]\ttrain_time={train_time:.1f}s\ttest_time={test_time:.1f}")

            stop = False
            for task in self.args.tasks:
                self.log(f"[step={step}]\t{task}_loss={all_loss[task]/self.args.eval_every:.2f}, "\
                        f"train_{task}_score={all_correct[task]}/{all_total[task]}={100*all_correct[task]/all_total[task]:.2f}")
                self.log(f"[step={step}]\tdev_{task}_score={100*res[f'{task}_score']:.2f}")
                all_loss[task] = all_correct[task] = all_total[task] = 0

                # score = sum(res[f'{task}_score'] for task in self.args.tasks)
                score = res[f'{self.args.tasks[0]}_score'] # only the first task

                if score > best_score:
                    best_score = score
                    self.save_model()
                    waited = 0
                # only start counting after reasonable number of steps
                elif step > 10000:
                    waited += 1
                    if waited > self.args.patience:
                        self.log('out of patience')
                        stop = True
            if stop or step >= self.args.max_step:
                break

        # TODO: see if makes any difference  
        self.log('BEFORE FINETUNE')
        self.load_model()
        t0 = time()
        res = self.predict_all(self.dev_sents, pipeline=True)
        for task in self.args.tasks:
            self.log(f"before_{task}_score={100*res[f'{task}_score']:.2f}, time={res[f'{task}_time']:.1f}s")
        best_scores = {t:res[f'{task}_score'] for t in self.args.tasks}
        self.finetune(best_scores)

        self.load_model()

        self.log('FINAL DEV')
        res = self.predict_all(self.dev_sents, pipeline=False)
        for task in self.args.tasks:
            self.log(f"best_{task}_score={100*res[f'{task}_score']:.2f}")

        if len(self.args.tasks) > 1:
            self.log('PIPELINE DEV')
            res = self.predict_all(self.dev_sents, pipeline=True)
            for task in self.args.tasks:
                self.log(f"pipeline_{task}_score={100*res[f'{task}_score']:.2f}")

        if self.test_sents:
            self.log('FINAL TEST')
            res = self.predict_all(self.test_sents, pipeline=True)
            for task in self.args.tasks:
                self.log(f"test_{task}_score={100*res[f'{task}_score']:.2f}")




    def finetune(self, best_scores):        
        # freeze all encoders
        self.encoders['feat'].set_freeze(True)

        for task in self.args.tasks:
            # self.decoders[task].tree_encoder.set_freeze(True)

            # restart the trainer to clear the momentum from the previous training
            self.trainer = dy.AdamTrainer(self.model)
            # load the best model from the previous finetuning 
            self.load_model()
            self.log(f'Start finetuning {task}')
            switch_trainer = (len(self.args.tasks) == 1) # directly change to SGD if there is only one task
            switched = False
            waited = 0
            step = 0

            for batch in iterate_batch(self.train_sents, self.args.eval_every):
                loss = total = correct = 0

                if switch_trainer:
                    self.trainer = dy.MomentumSGDTrainer(self.model)
                    switch_trainer = False
                    switched = True

                # train on a batch of sentences
                t0 = time()
                for sent in tqdm(batch):
                    step += 1
                    self.encode_sent(sent, True)
                    res = self.decoders[task].train_one_step(sent)
                    sent.clear_pred()
                    loss += res['loss']
                    total += res['total']
                    correct += res['correct']
                    if res['loss_expr']:
                        try:
                            res['loss_expr'].backward()
                            self.trainer.update()
                        except:
                            self.log('bad gradient, load previous model')
                            self.load_model()
                train_time = time() - t0


                # evaluate on dev set
                res = self.predict(self.dev_sents[:1000], task)
                score = res['score']

                self.log(f"[step={step}]\ttrain_time={train_time:.1f}s\ttest_time={res[f'time']:.1f}s")
                self.log(f"[step={step}]\t{task}_loss={loss/self.args.eval_every:.2f}, "\
                        f"train_{task}_score={correct}/{total}={100*correct/total:.2f}")
                self.log(f"[step={step}]\tdev_{task}_score={100*score:.2f}")

                if score > best_scores[task]:
                    best_scores[task] = score
                    self.save_model()
                    waited = 0
                else:
                    waited += 1
                    if waited > self.args.patience:
                        if switched:
                            self.log('out of patience')
                            break
                        else:
                            self.log('switch trainer')
                            switch_trainer = True
                            waited = 0

                if step >= self.args.max_step:
                    break

            self.log(f'Finish finetuning {task}')
            self.decoders[task].set_freeze(True)


    def predict(self, sents, task):
        # only for predicting single step, no pipeline, no output 
        res = defaultdict(float)
        for sent in tqdm(sents):
            sent.clear_pred()
            self.encode_sent(sent)
            t0 = time()
            self.decoders[task].predict(sent)
            res[f'time'] += (time()-t0)
        res[f'score'] = self.decoders[task].evaluate(sents)
        return res

    def predict_all(self, sents, pipeline=False, output=False):
        res = defaultdict(float)
        for sent in tqdm(sents):
            sent.clear_pred()
            self.encode_sent(sent)

            for task in self.args.tasks:
                t0 = time()
                self.decoders[task].predict(sent, pipeline)
                res[f'{task}_time'] += (time()-t0)

                if output:
                    if task == 'lin' or task == 'tsp' or task=='tsp-full':
                        reorder(sent, 'linearized_tokens')
                    elif task == 'gen':
                        reorder(sent, 'generated_tokens')
                    elif task == 'swap' or task == 'jump':
                        reorder(sent, 'sorted_tokens')
        for task in self.args.tasks:
            res[f'{task}_score'] = self.decoders[task].evaluate(sents)

        return res



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("mode", choices=['train', 'pred'])
    parser.add_argument("-m", "--model_file")
    parser.add_argument("-t", "--train_file")
    # parser.add_argument("-e", "--extra_file")
    parser.add_argument("-d", "--dev_file")
    parser.add_argument("-i", "--input_file")
    parser.add_argument("-p", "--pred_file")
    parser.add_argument("-g", "--gold_file")
    parser.add_argument("--tasks", default='tsp', help='combinations of: lin, tsp, tsp-full, swap, inf, gen, con')
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--max_step", type=int, default=1000000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--features", default='lemma+upos+label+morph')
    parser.add_argument("--pointer_type", default='glimpse', choices=['simple', 'glimpse', 'self'])
    parser.add_argument("--tree_lstm", default='simple', choices=['simple', 'att', 'selfatt'])
    parser.add_argument("--head_input", default='deps_vec', choices=['vec', 'deps_vec', 'deps_mem'])

    # beam linearizer
    parser.add_argument("--beam_size", type=int, default=16)
    parser.add_argument("--lin_decoders", default='h2d+l2r+r2l', help='combinations of: l2r, r2l, h2d')


    # WARNING: unstable, experimental settings,
    # keep them as default
    parser.add_argument("--tree_vecs", default='tree', help='combinations of: tree, seq, bag')
    parser.add_argument("--ignore_lemma_case", action='store_true')
    parser.add_argument("--lemmatize", action='store_true')
    # parser.add_argument("--avg_loss", action='store_true')
    parser.add_argument("--no_inf_rules", action='store_true')
    # parser.add_argument("--no_seq", action='store_true')
    parser.add_argument("--no_lin_constraint", action='store_true')
    parser.add_argument("--sent_tsp", action='store_true')
    parser.add_argument("--max_vocab", type=int, default=100000)
    # parser.add_argument("--extra_ratio", type=int, default=1)
    parser.add_argument("--first_train", type=int, default=1000000)
    parser.add_argument("--first_extra", type=int, default=1000000)
    parser.add_argument("--ud_train", action='store_true')
    parser.add_argument("--ud_dev", action='store_true')
    parser.add_argument("--ud_test", action='store_true')
    parser.add_argument("--tsp_update", default='all', choices=['row', 'col', 'all', 'path'])
    parser.add_argument("--guided_local_search", action='store_true')
    # parser.add_argument("--max_iter_sort", type=int, default=1)
    parser.add_argument("--orig_word", action='store_true')
    parser.add_argument("--pred_seq", action='store_true')
    parser.add_argument("--pred_tree", action='store_true')
    parser.add_argument("--no_xpos", action='store_true')


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
        # assert args.pred_file is not None 
        model = Realization(args)
        # args for prediction/eval
        model.args.guided_local_search = args.guided_local_search

        skip_lost = ('inf' not in model.args.tasks and 'con' not in model.args.tasks)
        print('skip_lost:', skip_lost)
        test_sents = list(read_conllu(args.input_file, args.ud_test, skip_lost, model.args.lemmatize, model.args.orig_word)) # lemma, no_word

        print('number of test sents:', len(test_sents))

        t0 = time()
        res = model.predict_all(test_sents, pipeline=len(model.args.tasks) > 1, output=True)
        print(f'TIME={time()-t0:.1f}s')
        for task in model.args.tasks:
            print(f"eval_{task}_score={100*res[task+'_score']:.2f}, time={res[task+'_time']:.1f}s")

        if args.pred_file:
            if 'con' in model.args.tasks:
                for sent in test_sents:
                    tokens = sent['linearized_tokens'] or sent['gold_linearized_tokens'] or sent.get_tokens()
                    capitalize(tokens, args.ignore_lemma_case)
                write_txt(args.pred_file, test_sents)
            else:
                write_conllu(args.pred_file, test_sents, ud=args.ud_test, use_morphstr=True)
    else:
        print('wrong mode!')

