We have modified the decoding GRU cell to attend to previous states (generated word predictions) in each step

<img src="https://github.com/zhedongzheng/finch/blob/master/assets/lookback_rnn.jpg" height='300'>

---
``` python train.py ```
```
Step 23429 | [30/30] | [750/781] | nll_loss:49.1 | kl_w:1.000 | kl_loss:6.35 

G: at the bullfight scene where to begin with this film as much as i can <end>
------------
I: the 60´s is a well balanced mini series between historical facts and a good plot

D: <start> the 60´s is a well <unk> mini series between <unk> facts and <unk> good <unk>

O: the bfg is one of the best movies of all time in the history of <end>
------------
I: i love this film and i think it is one of the best films

O: i absolutely loved this film i have never been a fan of the tv series <end>
------------
I: this movie is a waste of time and there is no point to watch it

O: this is a very good movie and it is not as bad as it is <end>
------------
```
where:
* I is the encoder input

* D is the decoder input (if 90% word dropout is set, then about 9 out of 10 words are unknown)

* O is the decoder output with regards to encoder input I

* G is random text generation, replacing the latent vector (z) by unit gaussian
---