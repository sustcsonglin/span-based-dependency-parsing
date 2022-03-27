# span-based-dependency-parsing
Source code of ACL2022 "[Headed-Span-Based Projective Dependency Parsing](http://arxiv.org/abs/2108.04750)" and 
Findings of ACL2022"[Combining (second-order) graph-based and headed-span-based projective dependency parsing](https://arxiv.org/pdf/2108.05838.pdf)"

## Setup
setup environment 
```
conda create -n parsing python=3.7
conda activate parsing
while read requirement; do pip install $requirement; done < requirements.txt 
```

setup dataset:

you can download the datasets I used from [link](https://mega.nz/file/jFIijLTI#b0b7550tdYVNcpGfgaXc0sk0F943lrt8D35v1SW2wbg). 

# Run
```
python train.py +exp=base  datamodule=a model=b seed=0
a={ptb, ctb, ud2.2}
b={biaffine, biaffine2o, span, span1o, span1oheadsplit, span2oheadsplit}
```

multirun example:
```
python train.py +exp=base datamodule.ud2.2 model=b datamodule.ud_lan=de,it,en,ca,cs,es,fr,no,ru,es,nl,bg seed=0,1,2 --mutlirun
```
For UD, you also need to setup the JAVA environment for the use of MaltParser. 
You need download MaltParser v1.9.2 from [link](https://www.maltparser.org/download.html). 

# Contact
Please let me know if there are any bugs. Also, feel free to contact bestsonta@gmail.com if you have any questions.

# Citation
```
@inproceedings{yang-tu-2022-headed,
  title={Headed-Span-Based Projective Dependency Parsing},
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
  author={Songlin Yang and Kewei Tu},
  year={2022}
}

@misc{yang-tu-2022-combining,
      title={Combining (second-order) graph-based and headed-span-based projective dependency parsing}, 
      author={Songlin Yang and Kewei Tu},
      year={2022},
    booktitle = "Findings of ACL",
}
```

# Acknowledge
The code is based on [lightning+hydra](https://github.com/ashleve/lightning-hydra-template) template. I use [FastNLP](https://github.com/fastnlp/fastNLP) as dataloader. I use lots of built-in modules (LSTMs, Biaffines, Triaffines, Dropout Layers, etc) from [Supar](https://github.com/yzhangcs/parser/tree/main/supar).  

