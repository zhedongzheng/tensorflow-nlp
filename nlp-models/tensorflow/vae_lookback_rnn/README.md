We have modified the decoding GRU cell to attend to previous states (generated word predictions) in each step

<img src="https://github.com/zhedongzheng/finch/blob/master/assets/lookback_rnn.jpg" height='300'>

``` python train.py ```
```
Step 23429 | [30/30] | [750/781] | nll_loss:50.5 | kl_w:1.000 | kl_loss:14.03 

G: this is the worst movie ever made the film i have ever seen in the <end>
------------
I: the 60´s is a well balanced mini series between historical facts and a good plot

D: <start> <unk> <unk> <unk> <unk> <unk> balanced <unk> <unk> <unk> historical <unk> <unk> a <unk> plot

O: the movie is one of the most interesting and the actors and a great cast <end>
------------
I: i love this film and i think it is one of the best films

O: i love this movie and i found it was a fan of the best films <end>
------------
I: this movie is a waste of time and there is no point to watch it

O: this movie is a complete waste of time to this movie and don't miss it <end>
------------
Model saved in file: ./saved/vrae.ckpt
```
where:
* I is the encoder input

* D is the decoder input (if 90% word dropout is set, then about 9 out of 10 words are unknown)

* O is the decoder output with regards to encoder input I

* G is random text generation, replacing the latent vector (z) by unit gaussian

``` python test.py ```
```
Loading trained model ...
I: i love this firm and it is beyond my expectation

O: i watched this movie because it was very much more than it at all costs <end>
------------
I: i want to watch this movie again because it is so interesting

O: i can to say that this is no reason why this movie was so bad <end>
------------
I: the time taken to develop the characters is quite long

O: the other people are so many people like this is the film to see it <end>
------------
I: is there any point to make a bad movie like this

O: to watch this film you have a lot but it's a good movie that is <end>
------------
I: sorry but there is no point to watch this movie again

O: movie but it was so bad i mean the people have seen this movie again <end>
------------
I: to be honest this movie is not worth my time and money

O: don't bother with this movie is so bad if you're on imdb com and enjoy <end>
```
---
Reference
* [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349)
