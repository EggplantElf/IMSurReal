# IMSurReal

**[v1.0]**: IMS Surface Realization System for INLG2019 and SR'19 Shared Task, see (http://taln.upf.edu/pages/msr2019-ws/SRST.html) for the details of the task.

**[v2.0]**: Improved linearization decoder (based on the Traveling Salesman Problem, see the ACL2020 paper for details) with major code refactorization.
  
## Required Libraries

* Python3
* DyNet2.0 (CPU version: `pip install dynet`)
* NumPy (`pip install numpy`)
* Levinshtein (`pip install python-Levenshtein`)
* mosestokenizer (`pip install mosestokenizer`)
* NLTK (`pip install nltk`)
* OR-Tools (`pip install ortools`)

## Usage

An example script containing all the commands is provided in `run_all.sh`, and the detailed usage is described as follows:

### Preparing data

This step aligns the original UD treebank with the T1 and T2 training and development data, so that the data has information about the original_id, and in the T2 sets the removed token are shown (with a '<LOST>' label in the dependency relation)

Note: The information about the original id and removed tokens are not used by the model, it is just easier for training and evaluation.

* Align the data for the shallow track (T1):
`python IMSurReal/align.py data/UD/en_partut-ud-train.conllu data/T1/en_partut-ud-train.conllu data/T1/en_partut-ud-train.gold.conllu`

* Align the data for the deep track (T2):
`python IMSurReal/align.py data/UD/en_partut-ud-train.conllu data/T2/en_partut-ud-train_DEEP.conllu data/T2/en_partut-ud-train_DEEP.gold.conllu`
  
### Training models

Train a model for each task:

* [v1.0]: linearize the unordered dependency tree (lin)
* [v2.0]: linearize the unordered dependency tree with TSP decoder (tsp)
* [v2.0]: post-processing with swap transition system to produce non-projective sentences (swap)
* generate missing function words for T2 (gen)
* inflect lemma into word form (inf)
* contract several tokens into one token (con)

>
`python IMSurReal/main.py train -m [MODEL_FILE] -t [TRAIN_FILE] -d [DEV_FILE] --tasks {lin, tsp, swap, gen, inf, con}`

where `[TRAIN_FILE]` and `[DEV_FILE]` should be the output from `align.py`, i.e., annotated with the information about the original_id and removed tokens.

Note that although it is possible to train several tasks in one model, e.g. use the flag `--tasks lin+inf` instead of `--tasks lin`, it generally performs worse than separated models, thus not recommended. 
  
### Prediction and/or Evaluation

`python IMSurReal/main.py pred -m [MODEL_FILE] -i [INPUT_FILE] -p [PRED_FILE]`

It outputs and prediction file, which could be used as the input file for the next step in the pipeline. The input file could be dev data or test data, where the alignment is not used. If no prediction file (-p) is provided, it just returns the evaluation score (Accuracy for inflection, BLEU for all other tasks).

### Detokenization

`python IMSurReal/detokenize.py [INPUT_FILE] [OUTPUT_FILE] [LANG]`

It creates the detokenized text output with Moses detokenizer for the human evaluation of the shared task.
  
#### Citation

If you are using this system for research, we would appreciate if you cite one of the follow papers:

* For the linearization module and the (non-official) results in the shallow track of the SR'18 Shared Task data:
>
@inproceedings{yu-etal-2019-head,
    title = "Head-First Linearization with Tree-Structured Representation",
    author = "Yu, Xiang  and
      Falenska, Agnieszka  and
      Vu, Ngoc Thang  and
      Kuhn, Jonas",
    booktitle = "Proceedings of the 12th International Conference on Natural Language Generation",
    month = oct # "{--}" # nov,
    year = "2019",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-8636",
    doi = "10.18653/v1/W19-8636",
    pages = "279--289"
}
  
* For the full pipeline and the results in both shallow and deep tracks of the SR'19 Shared Task:
>
@inproceedings{yu-etal-2019-imsurreal,
    title = "{IMS}ur{R}eal: {IMS} at the Surface Realization Shared Task 2019",
    author = "Yu, Xiang  and
      Falenska, Agnieszka  and
      Haid, Marina  and
      Vu, Ngoc Thang  and
      Kuhn, Jonas",
    booktitle = "Proceedings of the 2nd Workshop on Multilingual Surface Realisation (MSR 2019)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-6306",
    doi = "10.18653/v1/D19-6306",
    pages = "50--58"
}

* For the improved TSP-based linearization decoder and non-projective post processing (in v2.0):
>
@inproceedings{yu-etal-2020-fast,
    title = "Fast and Accurate Non-Projective Dependency Tree Linearization",
    author = "Yu, Xiang  and
      Tannert, Simon  and
      Vu, Ngoc Thang  and
      Kuhn, Jonas",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.134",
    doi = "10.18653/v1/2020.acl-main.134",
    pages = "1451--1462"
}
