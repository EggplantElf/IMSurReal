import sys
from collections import defaultdict

def read_conllu(filename):
    data = []
    text = ''
    sent = ''
    for line in open(filename):
        if line.strip():
            if line.startswith('# text = '):
                sent = line.strip().split('# text = ')[1]
            text += line 
        else:
            data.append((sent, text))
            text = ''
            sent = ''
    return data



def ensemble(output_file, *input_files):
    all_data = [read_conllu(input_file) for input_file in input_files]

    with open(output_file, 'w') as out:
        for hyps in zip(*all_data):
            counts = defaultdict(int)
            sent2text = {}
            for (sent, text) in hyps:
                sent2text[sent] = text
                counts[sent] += 1
            best_sent = max(counts.items(), key=lambda x:x[1])[0]
            best_text = sent2text[best_sent]
            out.write(best_text + '\n')


if __name__ == '__main__':
    ensemble(sys.argv[1], *sys.argv[2:])