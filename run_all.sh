#!/bin/sh
set -ue

# LANG=en_partut
LANG=fr_partut

T1_TRAIN=data/$LANG/T1/$LANG-ud-train.gold.conllu
T1_DEV=data/$LANG/T1/$LANG-ud-dev.gold.conllu
T2_TRAIN=data/$LANG/T2/$LANG-ud-train_DEEP.gold.conllu
T2_DEV=data/$LANG/T2/$LANG-ud-dev_DEEP.gold.conllu

T1_TEST=data/$LANG/T1/$LANG-ud-test.conllu
T2_TEST=data/$LANG/T2/$LANG-ud-test_DEEP.conllu

DIR=example/$LANG
mkdir -p $DIR


# Prepare train and dev data
# T1
python IMSurReal/align.py data/$LANG/UD/$LANG-ud-train.conllu data/$LANG/T1/$LANG-ud-train.conllu $T1_TRAIN
python IMSurReal/align.py data/$LANG/UD/$LANG-ud-dev.conllu data/$LANG/T1/$LANG-ud-dev.conllu $T1_DEV
# T2
python IMSurReal/align.py data/$LANG/UD/$LANG-ud-train.conllu data/$LANG/T2/$LANG-ud-train_DEEP.conllu $T2_TRAIN
python IMSurReal/align.py data/$LANG/UD/$LANG-ud-dev.conllu data/$LANG/T2/$LANG-ud-dev_DEEP.conllu $T2_DEV


# Train T1
# linearization with TSP decoder
python IMSurReal/main.py train -m $DIR/$LANG.t1.tsp.mdl -t $T1_TRAIN --d $T1_DEV --task tsp --max_step 4000
# (optional) swap post-processing, for treebanks with many non-projective trees
python IMSurReal/main.py train -m $DIR/$LANG.t1.swap.mdl -t $T1_TRAIN --d $T1_DEV --task swap --max_step 4000
# inflection
python IMSurReal/main.py train -m $DIR/$LANG.t1.inf.mdl -t $T1_TRAIN --d $T1_DEV --task inf --max_step 4000
# (optional) contraction, for some treebanks with contracted tokens
python IMSurReal/main.py train -m $DIR/$LANG.t1.con.mdl -t $T1_TRAIN --d $T1_DEV --task con --max_step 4000

# Train T2
# linearization with TSP decoder
python IMSurReal/main.py train -m $DIR/$LANG.t2.tsp.mdl -t $T2_TRAIN --d $T2_DEV --task tsp --max_step 4000
# (optional) swap post-processing, for treebanks with many non-projective trees
python IMSurReal/main.py train -m $DIR/$LANG.t2.swap.mdl -t $T2_TRAIN --d $T2_DEV --task swap --max_step 4000
# completion (function words generation)
python IMSurReal/main.py train -m $DIR/$LANG.t2.gen.mdl -t $T2_TRAIN --d $T2_DEV --task gen --max_step 4000
# inflection
python IMSurReal/main.py train -m $DIR/$LANG.t2.inf.mdl -t $T2_TRAIN --d $T2_DEV --task inf --max_step 4000
# (optional) contraction, for some treebanks with contracted tokens
python IMSurReal/main.py train -m $DIR/$LANG.t2.con.mdl -t $T2_TRAIN --d $T2_DEV --task con --max_step 4000


# Predict on test data
# T1
python IMSurReal/main.py pred  -m $DIR/$LANG.t1.tsp.mdl -i $T1_TEST -p $DIR/$LANG.t1.tsp.conllu
python IMSurReal/main.py pred  -m $DIR/$LANG.t1.swap.mdl -i $DIR/$LANG.t1.tsp.conllu -p $DIR/$LANG.t1.swap.conllu
python IMSurReal/main.py pred  -m $DIR/$LANG.t1.inf.mdl -i $DIR/$LANG.t1.swap.conllu -p $DIR/$LANG.t1.inf.conllu
python IMSurReal/main.py pred  -m $DIR/$LANG.t1.con.mdl -i $DIR/$LANG.t1.inf.conllu -p $DIR/$LANG.t1.con.conllu


# T2
python IMSurReal/main.py pred  -m $DIR/$LANG.t2.tsp.mdl -i $T2_TEST -p $DIR/$LANG.t2.tsp.conllu
python IMSurReal/main.py pred  -m $DIR/$LANG.t2.swap.mdl -i $DIR/$LANG.t2.tsp.conllu -p $DIR/$LANG.t2.swap.conllu
python IMSurReal/main.py pred  -m $DIR/$LANG.t2.gen.mdl -i $DIR/$LANG.t2.swap.conllu -p $DIR/$LANG.t2.gen.conllu
python IMSurReal/main.py pred  -m $DIR/$LANG.t2.inf.mdl -i $DIR/$LANG.t2.gen.conllu -p $DIR/$LANG.t2.inf.conllu
python IMSurReal/main.py pred  -m $DIR/$LANG.t2.con.mdl -i $DIR/$LANG.t2.inf.conllu -p $DIR/$LANG.t2.con.conllu

# evaluate test prediction (BLEU score on tokenized text)
python IMSurReal/evaluate.py data/$LANG/UD/$LANG-ud-test.conllu $DIR/$LANG.t1.inf.conllu
python IMSurReal/evaluate.py data/$LANG/UD/$LANG-ud-test.conllu $DIR/$LANG.t2.inf.conllu
