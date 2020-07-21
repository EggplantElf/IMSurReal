# IMSurReal
IMS Surface Realization System for INLG2019 and SR'19 Shared Task, see (http://taln.upf.edu/pages/msr2019-ws/SRST.html) for the details of the task.

#### Required Libraries
* Python3
* DyNet2.0 (CPU version: `pip install dynet`)
* NumPy (`pip install numpy`)
* Levinshtein (`pip install python-Levenshtein`)
* mosestokenizer (`pip install mosestokenizer`)
* NLTK (`pip install nltk`)

#### Usage
An example script containing all the commands is provided in `run_all.sh`, and the detailed usage is described as follows:

##### Preparing data
This step aligns the original UD treebank with the T1 and T2 training and development data, so that the data has information about the original_id, and in the T2 sets the removed token are shown (with a '<LOST>' label in the dependency relation)
Note: The information about the original id and removed tokens are not used by the model, it is just easier for training and evaluation.

* Align the data for the shallow track (T1): 
`python IMSurReal/align.py data/UD/en_partut-ud-train.conllu data/T1/en_partut-ud-train.conllu data/T1/en_partut-ud-train.gold.conllu`

* Align the data for the deep track (T2): 
`python IMSurReal/align.py data/UD/en_partut-ud-train.conllu data/T2/en_partut-ud-train_DEEP.conllu data/T2/en_partut-ud-train_DEEP.gold.conllu`


##### Training models
Train a model for each task: 
* linearize the unordered dependency tree (lin)
* generate missing function words for T2 (gen)
* inflect lemma into word form (inf)
* contract several tokens into one token (con)
> 

`python IMSurReal/main.py train -m [MODEL_FILE] -t [TRAIN_FILE] -d [DEV_FILE] --tasks {lin, gen, inf, con}`
where `[TRAIN_FILE]` and `[DEV_FILE]` should be the output from `align.py`, i.e., annotated with the information about the original_id and removed tokens.

Note that although it is possible to train several tasks in one model, e.g. use the flag `--tasks lin+inf` instead of `--tasks lin`, it generally performs worse than separated models, thus not recommended. 

##### Evaluation
`python IMSurReal/main.py eval -m [MODEL_FILE] -i [DEV_FILE]`
There is no output file in eval mode, just the evaluation score (accuracy or BLEU score, depending on the task).

##### Prediction
`python IMSurReal/main.py pred -m [MODEL_FILE] -i [INPUT_FILE] -p [PRED_FILE]`
It outputs and prediction file, which could be used as the input file for the next step in the pipeline. The input file could be dev data or test data, where the alignment is not used.


##### Detokenization
`python IMSurReal/detokenize.py [INPUT_FILE] [OUTPUT_FILE] [LANG]`


#### Citation:

If you are using this system for research, we would appreciate if you cite one of the follow papers:
* For the linearization module and the (non-official) results in the shallow track of the SR'18 Shared Task data:
>
    @inproceedings{yu2019head,
        title = "Head-First Linearization with Tree-Structured Representation",
        author = "Xiang Yu and Agnieszka Falenska and Ngoc Thang Vu and Jonas Kuhn",
        booktitle = "Proceedings of the 12th International Conference on Natural Language Generation",
        year = "2019",
        address="Tokyo, Japan",
        url="https://www.inlg2019.com/assets/papers/147_Paper.pdf"
    }

* For the full pipeline and the results in both shallow and deep tracks of the SR'19 Shared Task:
>
    @inproceedings{yu2019imsurreal,
        title = "IMSurReal: IMS at the Surface Realization Shared Task 2019",
        author = "Xiang Yu and Agnieszka Falenska and Marina Haid and Ngoc Thang Vu and Jonas Kuhn",
        booktitle = "Proceedings of the Second Workshop on Multilingual Surface Realization",
        year = "2019",
        address="Hong Kong, China",
        url="https://www.aclweb.org/anthology/D19-6306/"
    }    
