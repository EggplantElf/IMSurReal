import dynet as dy
import dynet_modules as dm
import numpy as np
from utils import *



class BagEncoder(Encoder):
    def __init__(self, args, model, name='bag'):
        super().__init__(args, model)
        self.name = name
        self.attention = dm.Attention(self.model, self.args.token_dim, self.args.token_dim)
        self.log(f'Initialized <{self.__class__.__name__}>, params = {self.model.parameter_count()}')


    def encode(self, sent):
        input_mat = dy.concatenate_cols([t.vecs['feat'] for t in sent.tokens])
        output_mat = self.attention.encode(input_mat)
        for t,v in zip(sent.tokens, dy.transpose(output_mat)):
            t.vecs[self.name] = v
