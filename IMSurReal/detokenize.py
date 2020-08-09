import sys
from data import *
from mosestokenizer import MosesDetokenizer


def main(test_file, output_file, lang):
    lang = lang.split('_')[0]
    # detokenizer = MosesDetokenizer(lang)
    detokenize = MosesDetokenizer(lang)
    sent_id = 1
    with open(output_file, 'w+') as out:
        for line in open(test_file):
            if line.startswith('# text ='):
                line = line.split('=', 1)[1].strip()
                words = line.split()
                out_text = detokenize(words)
                out.write(f'# sent_id = {sent_id}\n# text = {out_text}\n\n')
                sent_id += 1

if __name__ == '__main__':
    main(*sys.argv[1:])