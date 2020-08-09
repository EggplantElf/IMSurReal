import dynet as dy
import numpy as np

def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    # print (output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05/(output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI**2 / 2)
            Q2 = Q**2
            Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        # print('Orthogonal pretrainer loss: %.2e' % loss)
        pass
    else:
        # print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))

def set_orthonormal(p):
    p.set_value(orthonormal_initializer(p.shape()[0], p.shape[1]))

def clip_all(model, low=-1, high=1):
    for p in model.parameter_list():
        p.clip_inplace(low, high)

def leaky_relu(x):
    return dy.bmax(.1*x, x)


# class MLP(object):
#     def __init__(self, model, di, dh, do):
#         self.model = model
#         self.bh = model.add_parameters(dh)
#         self.wh = model.add_parameters((dh, di))
#         self.bo = model.add_parameters(do)
#         self.wo = model.add_parameters((do, dh))

#     def forward(self, x):
#         h = dy.affine_transform([self.bh, self.wh, x])
#         o = dy.affine_transform([self.bo, self.wo, dy.tanh(h)])
#         return o

class MLP(object):
    def __init__(self, model, di, do, *dhs):
        self.model = model
        ds = [di] + list(dhs) + [do]
        self.bs = [model.add_parameters(d) for d in ds[1:]]
        self.ws = [model.add_parameters((j, i)) for i, j in zip(ds, ds[1:])]


    def forward(self, x):
        for i, (b, w) in enumerate(zip(self.bs, self.ws)):
            if i:
                x = leaky_relu(x)
            x = dy.affine_transform([b, w, x])
        return x

class TreeLSTM(object):
    def __init__(self, model, dm, att_type):
        self.model = model 
        self.att_type = att_type
        self.WS = [self.model.add_parameters((dm, dm), init=orthonormal_initializer(dm, dm)) for _ in "iouf"]
        self.US = [self.model.add_parameters((dm, dm), init=orthonormal_initializer(dm, dm)) for _ in "iouf"]
        self.BS = [self.model.add_parameters(dm) for _ in "iouf"]

        if self.att_type == 'att' or self.att_type == 'selfatt':
            self.attention = Attention(model, dm, dm)
        if self.att_type == 'selfatt':
            self.self_attention = Attention(model, dm, dm)


    def state(self, x, hs=None, cs=None):
        if not hs:
            # initial state
            Wi, Wo, Wu, Wf = self.WS
            bi, bo, bu, bf = self.BS

            i = dy.logistic(dy.affine_transform([bi, Wi, x]))
            o = dy.logistic(dy.affine_transform([bo, Wo, x]))
            u = dy.tanh(dy.affine_transform([bu, Wu, x]))
            c = dy.cmult(i, u)
            h = dy.cmult(o, dy.tanh(c))
            return h, c
        else:
            # transduce
            Ui, Uo, Uu, Uf = self.US
            bi, bo, bu, bf = self.BS
            Wi, Wo, Wu, Wf = self.WS

            if self.att_type == 'selfatt':
                hm = dy.concatenate_cols(hs)
                hm = self.self_attention.encode(hm)
                hm = self.attention.encode(hm, x)
            elif self.att_type == 'att':
                hm = dy.concatenate_cols(hs)
                hm = self.attention.encode(hm, x)
            else:
                hm = dy.esum(hs)

            i = dy.logistic(dy.affine_transform([bi, Ui, hm, Wi, x]))
            o = dy.logistic(dy.affine_transform([bo, Uo, hm, Wo, x]))
            u = dy.tanh(dy.affine_transform([bu, Uu, hm, Wu, x]))
            fs = [dy.logistic(dy.affine_transform([bf, Uf, h, Wf, x])) for h in hs]
            c_out = dy.cmult(i, u) + dy.esum([dy.cmult(f, c) for f, c in zip(fs, cs)])
            h_out = dy.cmult(o, dy.tanh(c_out))
            return h_out, c_out


class Attention(object):
    """
    A module for both self attention and normal attention with key, query and value
    """
    def __init__(self, model, dm, dk, dq=None):
        # dm = memory dimension
        # dk = key dimension
        # dq = query dimension (None for self-attention)
        dq = dq or dm
        self.w_q = model.add_parameters((dk, dq), init=orthonormal_initializer(dk, dq)) 
        self.w_k = model.add_parameters((dk, dm), init=orthonormal_initializer(dk, dm))
        self.w_v = model.add_parameters((dk, dm), init=orthonormal_initializer(dk, dm))
        self.factor = dk ** 0.5

    def encode(self, memory, query=None):
        query = query or memory # if no query then self attention
        Q = self.w_q * query
        K = self.w_k * memory
        V = self.w_v * memory
        A = dy.softmax(dy.transpose(K) * Q / self.factor)
        out = V * A
        return out


class GlimpsePointer:
    def __init__(self, model, token_dim, query_dim=None):
        self.model = model
        query_dim = query_dim or token_dim
        self.att_q = self.model.add_parameters((token_dim, query_dim), 
                init=orthonormal_initializer(token_dim, query_dim))
        self.att_q2 = self.model.add_parameters((token_dim, token_dim + query_dim), 
                init=orthonormal_initializer(token_dim, token_dim+query_dim))

    def point(self, seq_vec, cand_mat):
        # combined glimpse and attend
        cand_mat_trans = dy.transpose(cand_mat)
        a = dy.softmax(cand_mat_trans * (self.att_q * seq_vec))
        cand_vec = cand_mat * a
        q = dy.concatenate([seq_vec, cand_vec])
        s = cand_mat_trans * (self.att_q2 * q)
        return s

    def glimpse(self, seq_vec, cand_mat, cand_mat_trans):
        a = dy.softmax(cand_mat_trans * (self.att_q * seq_vec))
        cand_vec = cand_mat * a
        return cand_vec

    def attend(self, seq_vec, cand_mat_trans):
        # assume the seq_vec is concatenated with cand_vec
        s = cand_mat_trans * (self.att_q2 * seq_vec)
        return s


class SimplePointer:
    def __init__(self, model, token_dim, query_dim=None):
        self.model = model
        query_dim = query_dim or token_dim
        self.att_q = self.model.add_parameters((token_dim, query_dim), 
                init=orthonormal_initializer(token_dim, query_dim))

    def point(self, seq_vec, cand_mat):
        cand_mat_trans = dy.transpose(cand_mat)
        s = cand_mat_trans * (self.att_q * seq_vec)
        return s

class SelfPointer:
    def __init__(self, model, token_dim, query_dim=None):
        self.model = model
        query_dim = query_dim or token_dim
        self.att_q = self.model.add_parameters((token_dim, query_dim), 
                init=orthonormal_initializer(token_dim, query_dim))
        self.self_attention = Attention(self.model, token_dim, token_dim)

    def point(self, seq_vec, cand_mat):
        cand_mat = self.self_attention.encode(cand_mat)
        cand_mat_trans = dy.transpose(cand_mat)
        s = cand_mat_trans * (self.att_q * seq_vec)
        return s

class BiaffineAttention:
    def __init__(self, model, token_dim, hid_dim, self_attention=False):
        self.mlp_fr = MLP(model, token_dim, hid_dim)
        self.mlp_to = MLP(model, token_dim, hid_dim)
        self.attention_fr = Attention(model, token_dim, token_dim) if self_attention else None
        self.attention_to = Attention(model, token_dim, token_dim) if self_attention else None
        self.biaffine = model.add_parameters((hid_dim+1, hid_dim+1), 
                                # init=dy.ConstInitializer(0.))
                                init=orthonormal_initializer(hid_dim+1, hid_dim+1))

    def attend(self, fr_vecs, to_vecs):
        fr_mat = dy.concatenate_cols(fr_vecs)
        to_mat = dy.concatenate_cols(to_vecs)

        if self.attention_fr:
            fr_mat = self.attention_fr.encode(fr_mat)
            to_mat = self.attention_to.encode(to_mat)

        fr_mat = leaky_relu(self.mlp_fr.forward(fr_mat))
        fr_mat = dy.concatenate([fr_mat, dy.inputTensor(np.ones((1, len(fr_vecs)), dtype=np.float32))])
        to_mat = leaky_relu(self.mlp_to.forward(to_mat))
        to_mat = dy.concatenate([to_mat, dy.inputTensor(np.ones((1, len(to_vecs)), dtype=np.float32))])
        score_mat = dy.transpose(fr_mat) * self.biaffine * to_mat
        return score_mat

class BilinearAttention:
    def __init__(self, model, token_dim, hid_dim):
        self.mlp_fr = MLP(model, token_dim, hid_dim)
        self.mlp_to = MLP(model, token_dim, hid_dim)
        self.biaffine = model.add_parameters((hid_dim, hid_dim),
                                # init=dy.ConstInitializer(0.))
                                init=orthonormal_initializer(hid_dim, hid_dim))

    def attend(self, fr_vecs, to_vecs):
        fr_mat = dy.concatenate_cols(fr_vecs)
        to_mat = dy.concatenate_cols(to_vecs)
        score_mat = dy.transpose(leaky_relu(self.mlp_fr.forward(fr_mat))) * \
                     self.biaffine * \
                     leaky_relu(self.mlp_to.forward(to_mat))
        return score_mat

class PairAttention:
    def __init__(self, model, token_dim, hid_dim):
        self.mlp_fr = MLP(model, token_dim, hid_dim)
        self.mlp_to = MLP(model, token_dim, hid_dim)
        self.mlp_att = MLP(model, hid_dim*2, 1, hid_dim)

    def attend(self, fr_vecs, to_vecs):
        fr_vecs = [leaky_relu(self.mlp_fr.forward(f)) for f in fr_vecs]
        to_vecs = [leaky_relu(self.mlp_to.forward(t)) for t in to_vecs]
        score_mat = dy.concatenate_cols([dy.concatenate([self.mlp_att.forward(dy.concatenate([f, t]))
                                                         for t in to_vecs])
                                        for f in fr_vecs])
        return score_mat







