import dynet as dy
import dynet_modules as dm
import numpy as np
from utils import *



class TreeEncoder(Encoder):
    def __init__(self, args, model, key = 'tree'):
        super().__init__(args, model)
        self.key = key
        # if 'head' in self.args.tree_vecs:
        self.head_chain_encoder = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, self.model)
        # if 'deps' in self.args.tree_vecs:
        self.tree_lstm = dm.TreeLSTM(self.model, self.args.token_dim, self.args.tree_lstm)
        self.special = self.model.add_lookup_parameters((1, self.args.token_dim))

        self.log(f'Initialized <{self.__class__.__name__}>, params = {self.model.parameter_count()}')


    def encode(self, sent, pred=False):
        sent.root.vecs['feat'] = self.special[0]

        self.encode_deps(sent.root, pred)
        self.encode_head(sent.root, None, pred) 

        for token in sent.tokens:
            # token.vecs[self.key] = (token.vecs[self.key+'_head'] if 'head' in self.args.tree_vecs else 0) + \
                                   # (token.vecs[self.key+'_deps'] if 'deps' in self.args.tree_vecs else 0)
            token.vecs[self.key] = token.vecs[self.key+'_head'] + token.vecs[self.key+'_deps']



    def encode_deps(self, head, pred=False):
        # propagate information bottom up 
        for dep in (head['pdeps'] if pred else head['deps']):
            self.encode_deps(dep)

        if (head['pdeps'] if pred else head['deps']):
            hs = [dep.vecs[self.key+'_deps'] for dep in head['deps']]
            cs = [dep.vecs['deps_mem'] for dep in head['deps']]
            head.vecs[self.key+'_deps'], head.vecs['deps_mem'] = self.tree_lstm.state(head.vecs['feat'], hs, cs)
        else:
            head.vecs[self.key+'_deps'], head.vecs['deps_mem'] = self.tree_lstm.state(head.vecs['feat'])



    def encode_head(self, token, head=None, pred=False):
        head_state = head['head_state'] if head else self.head_chain_encoder.initial_state()
        if 'deps' in self.args.tree_vecs:
            # independent top-down pass
            if self.args.head_input == 'feat_vec':
                token['head_state'] = head_state.add_input(token.vecs['feat'])
            # use bottom-up lstm output
            elif self.args.head_input == 'deps_vec':
                token['head_state'] = head_state.add_input(token.vecs[self.key+'_deps'])
            # use bottom-up lstm hidden cell
            else:
                token['head_state'] = head_state.add_input(token.vecs['deps_mem'])
        else:
            token['head_state'] = head_state.add_input(token.vecs['feat'])
        token.vecs[self.key+'_head'] = token['head_state'].output()
        for dep in (token['pdeps'] if pred else token['deps']):
            self.encode_head(dep, token, pred)


