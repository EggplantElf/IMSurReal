import dynet as dy
import dynet_modules as dm
import numpy as np
from utils import *


class SeqEncoder(Encoder):
    def __init__(self, args, model, name='seq'):
        super().__init__(args, model)
        self.name = name
        self.token_seq_encoder = dy.BiRNNBuilder(1, self.args.token_dim, self.args.token_dim, self.model, dy.VanillaLSTMBuilder)
        self.special = self.model.add_lookup_parameters((1, self.args.token_dim))
        self.log(f'Initialized <{self.__class__.__name__}>, params = {self.model.parameter_count()}')



    def encode(self, sent, key='input_tokens'):
        # TODO: cover more situations (swap, gen?)
        # use key sent['input_tokens']
        # tokens = sent['gold_linearized_tokens'] or sent['linearized_tokens'] or sent.get_tokens()
        # vecs = self.token_seq_encoder.transduce([t['vec'] for t in tokens])
        vecs = self.token_seq_encoder.transduce([t.vecs['feat'] for t in sent[key]])
        for t, v in zip(sent[key], vecs):
            t.vecs[self.name] = v  
        sent.root.vecs[self.name] = self.special[0]