import sys
from data import read_conllu, write_conllu, iterate
import nltk.translate.bleu_score as bs



def main(gold_file, pred_file, gkey='word', pkey='word'):
    gold_sents = read_conllu(gold_file, True, False) # Read UD format, (id, word, lemma, ...) 
    pred_sents = read_conllu(pred_file, False, False) # SRST format (id, lemma, word, ...)

    all_ref = [ [[t[gkey].lower() for t in sent.get_tokens()]] for sent in gold_sents]
    all_hyp = [ [t[pkey].lower() for t in sent.get_tokens()] for sent in pred_sents]


    chencherry = bs.SmoothingFunction()
    bleu = bs.corpus_bleu(all_ref, all_hyp, smoothing_function=chencherry.method2)
    exact = sum(r[0] == h for r, h in zip(all_ref, all_hyp)) / len(all_ref)
    print(f'eval: bleu={bleu:.4f}, exact={exact:.4f}')
    # print(f'{100*bleu:.2f}')




if __name__ == '__main__':
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print('Wrong number few arguments.')
        print(str(sys.argv[0]), 'reference-file', 'prediction-file', '[reference-key = WORD/lemma]', '[prediction-key = WORD/lemma]')
        exit()