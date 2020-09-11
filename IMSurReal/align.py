import sys
sys.path.append(".")
from data import *
import random
import Levenshtein


def check_constraints(pairs):
    fdict = {k['tid']:v for k,v in pairs}
    for t, u in pairs:
        try:
            domain = sorted(t['domain'], key=lambda x: fdict[x['tid']]['tid'])
        except:
            print(t['domain'])
            exit()
        lins = [(0 if x is t else x['lin']) for x in domain if x['lin'] is not None or x is t]
        if lins and lins != sorted(lins):
            return False
    return True


def similar(t1, t2, relax=False):
    if relax:
        d = Levenshtein.distance(t1, t2)
        return d / len(t1) < 0.6
    else:
        return t1 == t2


def align(th, uh, relax=False, verbose=False):
    out_agenda = []
    sigs = set()
    i = 0

    if verbose:
        print('th, uh', th, uh)

    if not (similar(th['lemma'], uh['lemma'], relax) or uh['lemma'] in ['@card@', '@ord@']):
        return out_agenda
    else:
        agenda = [([], [td for td in th['deps'] if td.not_empty()], uh['deps'], [])]
        while agenda:
            i += 1
            if i > 100000:
                return []
            pair, trest, urest, cur_missing = agenda.pop()
            if verbose:
                print('agenda.pop', pair, trest, urest)
            if trest:
                t = trest[0]
                for u in urest:
                    if verbose:
                        print('align', t, u, relax)
                    for apair, missing in align(t, u, relax):
                        if verbose:
                            print('apair', apair)
                        new_pair = pair + apair + [(t, u)]
                        agenda.append((new_pair, trest[1:], [x for x in urest if x is not u], cur_missing + missing))
            # elif pair:
            else:
                new_missing = cur_missing + ([(th, urest)] if th['tid'] and urest else [])
                out_agenda.append((pair, new_missing))
    if verbose:
        print('out', out_agenda)
    return out_agenda


def mark(sent, pairs, missing):
    # to avoid a bug in write_conllu
    sent['input_tokens'] = []

    for t, u in pairs:
        if t['lin'] is not None:
            t['morph'].append(f'lin={t["lin"]:+d}')
        t['morph'].append(f'original_id={u["tid"]}')
        t['oword'] = u['oword']
        t['cword'] = u['cword']

    # add missing words
    for t, us in missing:
        for u in us:
            n = Token({ 'tid': len(sent.tokens),
                        'oword': u['oword'],
                        'cword': u['cword'],
                        'olemma': u['olemma'],
                        'lemma': u['lemma'],
                        'upos': u['upos'],
                        'xpos': '_',
                        'hid': t['tid'],
                        'original_id': u['tid'],
                        'morph': u['morph'] + [f"original_id={u['tid']}"],
                        'label': '<LOST>'
                        })
            sent.add_token(n)

def align_sent(tsent, usent):
    ulemmas = set(u['lemma'] for u in usent.get_tokens())
    relax = any(t['lemma'] not in ulemmas for t in tsent.get_tokens())
    agenda = align(tsent.root, usent.root, relax)
    if agenda:
        for pairs, missing in agenda:
            if check_constraints(pairs):
                mark(tsent, pairs, missing)
                return tsent


def main(UD_file, in_file, out_file, scramble=False):
    print(UD_file)
    udsents = read_conllu(UD_file, True)
    t2sents = read_conllu(in_file, False)

    sent_id = 0
    with open(out_file, 'w') as out:
        out_sents = []
        for usent, tsent in zip(udsents, t2sents):
            sent_id += 1
            out_sent = align_sent(tsent, usent)
            if out_sent:
                line = f'# sent_id = {sent_id}\n'
                sorted_tokens = sorted(out_sent.tokens[1:], key=lambda x: x['original_id'])
                line += '# lemmata = ' + ' '.join([t['lemma'] for t in sorted_tokens]) + '\n'
                # line += '# words = ' + ' '.join([t['word'] for t in sorted_tokens]) + '\n'
                for t in out_sent.get_output_tokens():
                    morphstr = '_' if t['morph'] == [] else \
                            '|'.join(m for m in sorted(t['morph'], key=str.swapcase))
                    line += f"{t['tid']}\t{t['oword']}\t{t['olemma']}\t{t['upos']}\t{t['xpos']}\t{morphstr}\t" \
                                f"{t['hid']}\t{t['label']}\t{t['cword']}\t{t['original_id'] or '_'}\n" 
                out_sents.append(line + '\n')
            else:
                print(tsent)
    
        if scramble:
            random.shuffle(out_sents)

        for sent in out_sents:
            out.write(sent)

    print(f'aligned {sent_id} sentences')

if __name__ == '__main__':
    scramble = (len(sys.argv) > 4 and sys.argv[4] == 'scramble')
    main(sys.argv[1], sys.argv[2], sys.argv[3], scramble)
