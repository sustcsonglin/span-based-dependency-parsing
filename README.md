# span-based-dependency-parsing
Source code of "[Headed Span-Based Projective Dependency Parsing](http://arxiv.org/abs/2108.04750)" and "[Combining (second-order) graph-based and headed span-based projective dependency parsing]"(https://arxiv.org/pdf/2108.05838.pdf)"

## Setup
prepare environment 
```
conda create -n parsing python=3.7
conda activate parsing
while read requirement; do pip install $requirement; done < requirements.txt 
```

prepare dataset:

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
For UD, you also need to prepare the JAVA environment for the use of MaltParser.

# TODO
- Clean code (e.g. add comments)
- Add eval.py, now we only support training from scratch. 
- Release pre-trained model.

# Contact
Please let me know if there are any bugs. Also, feel free to contact bestsonta@gmail.com if you have any questions.

# Citation
```
@misc{yang2021headed,
      title={Headed Span-Based Projective Dependency Parsing}, 
      author={Songlin Yang and Kewei Tu},
      year={2021},
      eprint={2108.04750},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{yang2021combining,
      title={Combining (second-order) graph-based and headed span-based projective dependency parsing}, 
      author={Songlin Yang and Kewei Tu},
      year={2021},
      eprint={2108.05838},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

# Acknowledge
The code is based on [lightning+hydra](https://github.com/ashleve/lightning-hydra-template) template. I use [FastNLP](https://github.com/fastnlp/fastNLP) as dataloader. I use lots of built-in modules (LSTMs, Biaffines, Triaffines, Dropout Layers, etc) from [Supar](https://github.com/yzhangcs/parser/tree/main/supar).  

