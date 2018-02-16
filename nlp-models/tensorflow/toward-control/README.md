---
Implementing the idea of ["Toward Controlled Generation of Text"](https://arxiv.org/abs/1703.00955)

---
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/control-vae.png" height='300'>

```
python train_base_vae.py
```
```
Step 18737 | Train VAE | [24/25] | [750/781] | nll_loss:48.7 | kl_w:1.00 | kl_loss:15.90

Word Dropout
Data Shuffled

G: they are all the other films they were trying to be afraid of a film <end>
------------
I: i love this film it is so good to watch

O: to say that this movie is the worst movie i would give it 0 10 <end>

R: i really enjoyed this movie and i didn't want to write a copy of it <end>
------------
```
```
python train_discriminator.py
```
```
------------
Step 22647 | Train Discriminator | [4/25] | [750/781]
	| clf_loss:36.01 | clf_acc:0.90 | L_u: 5.34

Step 22647 | Train Encoder | [4/25] | [750/781]
	| seq_loss:65.2 | kl_w:1.00 | kl_loss:11.48

Step 22647 | Train Generator | [4/25] | [750/781]
	| seq_loss:65.7 | kl_w:1.00 | kl_loss:11.48
	| temperature:0.25 | l_attr_z:1.94 | l_attr_c:0.80
------------
I: i love this film it is so good to watch

O: this is the best movie ever made for a long time i have ever seen <end>

R: after watching this movie i give it a chance to watch this movie is horrible <end>
------------
I: this movie is horrible and waste my time

O: a insult to punish your money on the script and the acting was hideously dull <end>

R: this movie is fun to anyone who appreciates great music is the best movie ever <end>
------------
```
where:
* I is the encoder input

* G is the decoder output with prior z

* O is the decoder output with posterior z

* R is the decoder output with posterior z, when the attribute c (e.g. sentiment) is reversed
