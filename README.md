# Constitutional Classifier

Authors:

Ellina Aleshina, Pavel Gurevich, Ilya Sharov

This project is dedicaed to discover and implement the ideas from Anthropic's [paper](https://arxiv.org/pdf/2501.18837) about constitutional classifiers.

For hypothesis testing we reduced the general approach to the one small problem. 

Our task for now is to train the classifier that bans any prompts about Quentin Tarantino (constitution's analogue). For this purpoce we

1. Generated synthetic prompts about Tarantino and other movie directors
2. Collected non-malicious prompts about different topics not related to cinema
3. Augmented the data using attacks programs implementation from `h4rm3l` package
4. Trained the model on this data
5. Tested it in different settings (see the report)

## Environment setup

To download all the dependences run the following command:

```
conda env create -f environment.yml
```

## Model inference

To run the model inference you should go to the `classifier_finetuning` directory, run `python inference.py` and input your prompt into the stdin.

If the prompt is related to Tarantino, output will be "Tarantino", or "Not Tarantino" otherwise. 

The output should be like in the code below:

```
(constitution) root@a7a2bfe37bf6:/workspace/constitutional_classifier/classifier_finetuning# python inference.py 
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
Please, input your query:
When was Tarantino born?
Tarantino
```
