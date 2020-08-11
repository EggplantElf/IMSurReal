#!/bin/sh
set -ue

T1_TRAIN=data/T1/en_partut-ud-train.gold.conllu
T1_DEV=data/T1/en_partut-ud-dev.gold.conllu
T2_TRAIN=data/T2/en_partut-ud-train_DEEP.gold.conllu
T2_DEV=data/T2/en_partut-ud-dev_DEEP.gold.conllu

T1_TEST=data/T1/en_partut-ud-test.conllu
T2_TEST=data/T2/en_partut-ud-test_DEEP.conllu

mkdir -p example

# Prepare train and dev data
# T1
python IMSurReal/align.py data/UD/en_partut-ud-train.conllu data/T1/en_partut-ud-train.conllu $T1_TRAIN
python IMSurReal/align.py data/UD/en_partut-ud-dev.conllu data/T1/en_partut-ud-dev.conllu $T1_DEV
# T2
python IMSurReal/align.py data/UD/en_partut-ud-train.conllu data/T2/en_partut-ud-train_DEEP.conllu $T2_TRAIN
python IMSurReal/align.py data/UD/en_partut-ud-dev.conllu data/T2/en_partut-ud-dev_DEEP.conllu $T2_DEV


# Train T1
# linearization with TSP decoder
python IMSurReal/main.py train -m example/t1.tsp.mdl -t $T1_TRAIN --d $T1_DEV --task tsp
# (optional) swap post-processing, for treebanks with many non-projective trees
python IMSurReal/main.py train -m example/t1.swap.mdl -t $T1_TRAIN --d $T1_DEV --task swap
# inflection
python IMSurReal/main.py train -m example/t1.inf.mdl -t $T1_TRAIN --d $T1_DEV --task inf
# (optional) contraction, for some treebanks with contracted tokens
python IMSurReal/main.py train -m example/t1.con.mdl -t $T1_TRAIN --d $T1_DEV --task con

# Train T2
# linearization with TSP decoder
python IMSurReal/main.py train -m example/t2.tsp.mdl -t $T2_TRAIN --d $T2_DEV --task tsp
# (optional) swap post-processing, for treebanks with many non-projective trees
python IMSurReal/main.py train -m example/t2.swap.mdl -t $T2_TRAIN --d $T2_DEV --task swap
# completion (function words generation)
python IMSurReal/main.py train -m example/t2.gen.mdl -t $T2_TRAIN --d $T2_DEV --task gen
# inflection
python IMSurReal/main.py train -m example/t2.inf.mdl -t $T2_TRAIN --d $T2_DEV --task inf
# (optional) contraction, for some treebanks with contracted tokens
python IMSurReal/main.py train -m example/t2.con.mdl -t $T2_TRAIN --d $T2_DEV --task con


# Predict on test data
# T1
python IMSurReal/main.py pred  -m example/t1.tsp.mdl -i $T1_TEST -p example/t1.tsp.conllu
python IMSurReal/main.py pred  -m example/t1.swap.mdl -i example/t1.tsp.conllu -p example/t1.swap.conllu
python IMSurReal/main.py pred  -m example/t1.inf.mdl -i example/t1.swap.conllu -p example/t1.inf.conllu
python IMSurReal/main.py pred  -m example/t1.con.mdl -i example/t1.inf.conllu -p example/t1.con.conllu


# T2
python IMSurReal/main.py pred  -m example/t2.tsp.mdl -i $T1_TEST -p example/t2.tsp.conllu
python IMSurReal/main.py pred  -m example/t2.swap.mdl -i example/t2.tsp.conllu -p example/t2.swap.conllu
python IMSurReal/main.py pred  -m example/t2.gen.mdl -i example/t2.tsp.conllu -p example/t2.gen.conllu
python IMSurReal/main.py pred  -m example/t2.inf.mdl -i example/t2.gen.conllu -p example/t2.inf.conllu
python IMSurReal/main.py pred  -m example/t2.con.mdl -i example/t2.inf.conllu -p example/t2.con.conllu

# evaluate test prediction (BLEU score on tokenized text)
python IMSurReal/evaluate.py data/UD/en_partut-ud-test.conllu example/t1.inf.conllu
python IMSurReal/evaluate.py data/UD/en_partut-ud-test.conllu example/t2.inf.conllu
