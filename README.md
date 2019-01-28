# Text-Summary

#### Implementation of the paper [**Learning to Encode Text as Human-Readable Summaries using Generative Adversarial Networks**][paper].

[view paper locally][./paper.pdf]

_please install the requirements using the following command:_

`pip install -r requirements.txt`



This is an implementation of the [paper][paper] using **Deep Q Learning**, a branch of **Reinforcement Learning**. 

Here's the brief version:

![short-version](./short_version.png)

In `basic_main.py`, the _Generator_, denoted **G**, the Discriminator, denoted **D**, and the Reconstructor, denoted **R**, are all `seq2seq` models. **G+R** is an **Auto Encoder Network**, while **G+D** is trained with **Inverse Reinforcement Learning**. Techniques in use: **Dueling DQN**.

In comparison with `basic_main.py`, `Actor-Critic.py` and `Q.py` uses more advanced techniques, and is a lot more efficient in terms of memory.

[paper]: http://speech.ee.ntu.edu.tw/~tlkagk/paper/learning-encode-text.pdf
