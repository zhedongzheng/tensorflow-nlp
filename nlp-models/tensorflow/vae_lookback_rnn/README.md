We have modified the decoding GRU cell, in order to attend to previous states stored in memory, in each step

<img src="https://github.com/zhedongzheng/finch/blob/master/assets/lookback_rnn.jpg" height='300'>

---
``` python train.py ```
```
Step 10135 | [13/30] | [750/781] | nll_loss:56.8 | kl_w:1.000 | kl_loss:12.90 

G: a lot of those movies this is a good movie a film to be missed <end>
------------
I: fight scenes br br a great fun film that adults and children alike will enjoy

D: <start> <unk> <unk> br <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>

O: spoilers spoilers br br a great horror film that is a must see it to <end>
------------
I: i love this film and i think it is one of the best films

O: i rented this movie when i thought it was a one of the best movies <end>
------------
I: this movie is a waste of time and there is no point to watch it

O: this movie is a waste of time if you like this movie to be warned <end>
------------
```
where:
* I is the encoder input

* D is the decoder input (if 90% word dropout is set, then about 9 out of 10 words are unknown)

* O is the decoder output with regards to encoder input I

* G is random text generation, replacing the latent vector (z) by unit gaussian
---
