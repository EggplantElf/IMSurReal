#!/bin/sh
set -ue

T1_TRAIN=data/T1/en_partut-ud-train.gold.conllu
T1_DEV=data/T1/en_partut-ud-dev.gold.conllu
T2_TRAIN=data/T2/en_partut-ud-train_DEEP.gold.conllu
T2_DEV=data/T2/en_partut-ud-dev_DEEP.gold.conllu

T1_TEST=data/T1/en_partut-ud-test.conllu
T2_TEST=data/T2/en_partut-ud-test_DEEP.conllu

mkdir example

# Prepare train and dev data
# T1
python imsurreal/align.py data/UD/en_partut-ud-train.conllu data/T1/en_partut-ud-train.conllu $T1_TRAIN
python imsurreal/align.py data/UD/en_partut-ud-dev.conllu data/T1/en_partut-ud-dev.conllu $T1_DEV
# T2
python imsurreal/align.py data/UD/en_partut-ud-train.conllu data/T2/en_partut-ud-train_DEEP.conllu $T2_TRAIN
python imsurreal/align.py data/UD/en_partut-ud-dev.conllu data/T2/en_partut-ud-dev_DEEP.conllu $T2_DEV


# Train T1
# linearization
python imsurreal/main.py train -m example/t1.lin.mdl -t $T1_TRAIN --d $T1_DEV --task lin
# inflection
python imsurreal/main.py train -m example/t1.inf.mdl -t $T1_TRAIN --d $T1_DEV --task inf
# (optional) contraction, only for treebanks with contractions
python imsurreal/main.py train -m example/t1.con.mdl -t $T1_TRAIN --d $T1_DEV --task con

# Train T2
# linearization
python imsurreal/main.py train -m example/t2.lin.mdl -t $T2_TRAIN --d $T2_DEV --task lin
# completion (generation)
python imsurreal/main.py train -m example/t2.gen.mdl -t $T2_TRAIN --d $T2_DEV --task gen
# inflection
python imsurreal/main.py train -m example/t2.inf.mdl -t $T2_TRAIN --d $T2_DEV --task inf
# (optional) only for treebanks with contractions
python imsurreal/main.py train -m example/t2.con.mdl -t $T2_TRAIN --d $T2_DEV --task con


# Evaluate on dev data
# T1
python imsurreal/main.py eval -m example/t1.lin.mdl -i $T1_DEV
python imsurreal/main.py eval -m example/t1.inf.mdl -i $T1_DEV
python imsurreal/main.py eval -m example/t1.con.mdl -i $T1_DEV
# T2
python imsurreal/main.py eval -m example/t2.lin.mdl -i $T2_DEV
python imsurreal/main.py eval -m example/t2.gen.mdl -i $T2_DEV
python imsurreal/main.py eval -m example/t2.inf.mdl -i $T2_DEV
python imsurreal/main.py eval -m example/t2.con.mdl -i $T2_DEV


# Predict on test data
# T1
python imsurreal/main.py pred  -m example/t1.lin.mdl -i $T1_TEST -p example/t1.lin.conllu
python imsurreal/main.py pred  -m example/t1.inf.mdl -i example/t1.lin.conllu -p example/t1.inf.conllu
python imsurreal/main.py pred  -m example/t1.con.mdl -i example/t1.inf.conllu -p example/t1.con.conllu
python imsurreal/detokenize.py example/t1.con.conllu example/t1.txt en

# T2
python imsurreal/main.py pred  -m example/t2.lin.mdl -i $T1_TEST -p example/t2.lin.conllu
python imsurreal/main.py pred  -m example/t2.gen.mdl -i example/t2.lin.conllu -p example/t2.gen.conllu
python imsurreal/main.py pred  -m example/t2.inf.mdl -i example/t2.gen.conllu -p example/t2.inf.conllu
python imsurreal/main.py pred  -m example/t2.con.mdl -i example/t2.inf.conllu -p example/t2.con.conllu
python imsurreal/detokenize.py example/t2.con.conllu t2.txt en


