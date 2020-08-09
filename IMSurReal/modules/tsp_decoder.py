from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import dynet as dy
import dynet_modules as dm
import numpy as np
import random
from utils import *
from data import flatten
from time import time
from modules.seq_encoder import SeqEncoder
from modules.bag_encoder import BagEncoder
from modules.tree_encoder import TreeEncoder

class TSPDecoder(Decoder):
    def __init__(self, args, model, full = False):
        super().__init__(args, model)
        self.train_input_key = 'input_tokens'
        self.train_output_key = 'gold_linearized_tokens'
        self.pred_input_key = 'input_tokens'
        self.pred_output_key = 'linearized_tokens'
        self.vec_key = 'tsp_vec'

        if 'seq' in self.args.tree_vecs:
            self.seq_encoder = SeqEncoder(self.args, self.model, 'tsp_seq')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder = BagEncoder(self.args, self.model, 'tsp_bag')
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder = TreeEncoder(self.args, self.model, 'tsp_tree')

        self.full = full
        self.special = self.model.add_lookup_parameters((2, self.args.token_dim))
        self.biaffine = dm.BiaffineAttention(self.model, self.args.token_dim, self.args.hid_dim)

        if not full:
            self.f_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, model)
            self.b_lstm = dy.VanillaLSTMBuilder(1, self.args.token_dim, self.args.token_dim, model)
        self.log(f'Initialized <{self.__class__.__name__}>, params = {self.model.parameter_count()}')

    def encode(self, sent):
        # encode
        if 'seq' in self.args.tree_vecs:
            self.seq_encoder.encode(sent, 'linearized_tokens' if self.args.pred_seq else 'gold_linearized_tokens')
        if 'bag' in self.args.tree_vecs:
            self.bag_encoder.encode(sent)
        if 'tree' in self.args.tree_vecs:
            self.tree_encoder.encode(sent, self.args.pred_tree)
        sum_vecs(sent, self.vec_key, ['feat', 'tsp_seq', 'tsp_bag', 'tsp_tree'])
        # print([t['lemma'] for t in sent['gold_linearized_tokens']])
        # print([t['lemma'] for t in sent.tokens])
        # exit()


    def decode(self, tokens, constraints=[], train_mode=False):
        loss = 0
        errs = []

        fr_vecs = [self.special[0]] + [t.vecs[self.vec_key] for t in tokens]
        to_vecs = [self.special[1]] + [t.vecs[self.vec_key] for t in tokens]
        score_mat = self.biaffine.attend(fr_vecs, to_vecs)
        scores = score_mat.npvalue()

        if train_mode:
            oids = [0] + [t['original_id'] for t in tokens]
            gold_path = np.argsort(oids).tolist() + [0]
            trans_mat = dy.transpose(score_mat)
            for i, j in zip(gold_path, gold_path[1:]):
                errs.append(dy.hinge(score_mat[i], j))
                errs.append(dy.hinge(trans_mat[j], i))
            if errs:
                loss = dy.average(errs)

        costs = (1000 * (scores.max() - scores)).astype(int).tolist()
        solution = solve_tsp(costs, constraints, self.args.guided_local_search) # first is best
        if not solution:
            # self.log('no solution, remove constraints')
            solution = solve_tsp(costs, [], self.args.guided_local_search)

        assert solution != []
        seq = [tokens[i-1] for i in solution[1:-1]]

        return {'loss': loss, 
                'seq': seq}


    def get_subtree_constraints(self, head):
        lin_order = [head['domain'].index(t)+1 for t in head['order']]
        constraints = list(zip(lin_order, lin_order[1:]))
        return constraints

    def get_tree_constraints(self, sent):
        constraints = []
        tokens = sent[self.pred_input_key]
        for token in tokens:
            lin_order = [tokens.index(t)+1 for t in token['order']]
            constraints += list(zip(lin_order, lin_order[1:]))
        return constraints

    def predict(self, sent, pipeline=False):
        self.encode(sent)

        if self.full:
            constraints = [] if self.args.no_lin_constraint else self.get_tree_constraints(sent)
            res = self.decode(sent[self.pred_input_key], constraints)
            sent['linearized_tokens'] = res['seq']
        else:
            for token in traverse_bottomup(sent.root):
                domain = ([token] + token['pdeps']) if self.args.pred_tree else token['domain']
                if len(domain) > 1:
                    constraints = [] if self.args.no_lin_constraint else self.get_subtree_constraints(token)
                    res = self.decode(domain, constraints)
                    token['linearized_domain'] = res['seq']
                    # add predicted sequential information
                    f_vec = self.f_lstm.initial_state().transduce([t.vecs[self.vec_key] for t in res['seq']])[-1]
                    b_vec = self.b_lstm.initial_state().transduce([t.vecs[self.vec_key] for t in res['seq'][::-1]])[-1]
                    token.vecs[self.vec_key] += (f_vec + b_vec)
                else:
                    token['linearized_domain'] = [token]

            sent['linearized_tokens'] = flatten(token, 'linearized_domain')


    def train_one_step(self, sent):
        total = correct = loss = 0
        t0 = time()

        self.encode(sent)
        
        if self.full:
            constraints = [] if self.args.no_lin_constraint else self.get_tree_constraints(sent)
            res = self.decode(sent[self.train_input_key], constraints, True)
            loss = res['loss']
            total += 1
            sent['linearized_tokens'] = res['seq']
            correct += int(sent['linearized_tokens'] == sent['gold_linearized_tokens'] )
        else:
            for token in traverse_bottomup(sent.root):
                domain = ([token] + token['pdeps']) if self.args.pred_tree else token['domain']
                if len(domain) > 1:
                    constraints = [] if self.args.no_lin_constraint else self.get_subtree_constraints(token)
                    res = self.decode(domain, constraints, True)
                    token['linearized_domain'] = res['seq']
                    loss += res['loss']
                    total += 1
                    correct += int(token['linearized_domain'] == token['gold_linearized_domain'])
                    # add predicted sequential information
                    f_vec = self.f_lstm.initial_state().transduce([t.vecs[self.vec_key] for t in res['seq']])[-1]
                    b_vec = self.b_lstm.initial_state().transduce([t.vecs[self.vec_key] for t in res['seq'][::-1]])[-1]
                    token.vecs[self.vec_key] += (f_vec + b_vec)
                else:
                    token['linearized_domain'] = [token]

            sent['linearized_tokens'] = flatten(token, 'linearized_domain')

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
        pred_bleu = eval_all(gold_seqs, pred_seqs)
        print([t['lemma'] for t in  gold_seqs[0]])
        return pred_bleu


def solve_tsp(costs, constraints=[], beam_size=1, gls=False):
    manager = pywrapcp.RoutingIndexManager(len(costs), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return costs[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    solver = routing.solver()

    # linear order constraints
    if constraints:
        order_callback_index = routing.RegisterUnaryTransitCallback(lambda x: 1) # always add 1
        routing.AddDimension(order_callback_index, 0, len(costs)+1, True, 'Order')
        order = routing.GetDimensionOrDie('Order')

        for i, j in constraints:
            solver.Add(order.CumulVar(i) < order.CumulVar(j))


    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 1
    search_parameters.solution_limit = 100
    search_parameters.log_search = False

    if gls:
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
        out = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            out.append(manager.IndexToNode(index))
            index = assignment.Value(routing.NextVar(index))
        out.append(manager.IndexToNode(index))
        return out
    else:
        return []

