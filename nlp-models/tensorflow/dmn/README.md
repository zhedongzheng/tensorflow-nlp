<img src="https://github.com/zhedongzheng/finch/blob/master/assets/dmn-details.png">

* Many functions are adapted from [Alex Barron's](https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow) work, thanks for him!
* Based on that:
    * We have used ```tf.map_fn``` to replace the Python for loop, independent of sequence length, which makes the model truly dynamic
    * We have added a decoder in the answer module for "talking"
    * We have reproduced ```AttentionGRUCell``` from official ```GRUCell``` from TF 1.4

```
python train.py
```
```
Data Shuffled
[10/10] | [0/156] | loss:0.049 | acc:0.992
[10/10] | [50/156] | loss:0.046 | acc:0.992
[10/10] | [100/156] | loss:0.015 | acc:0.992
[10/10] | [150/156] | loss:0.009 | acc:1.000
final testing accuracy: 0.997

[['Fred', 'picked', 'up', 'the', 'football', 'there', '<end>'],
 ['Fred', 'gave', 'the', 'football', 'to', 'Jeff', '<end>'],
 ['Bill', 'went', 'back', 'to', 'the', 'bathroom', '<end>'],
 ['Jeff', 'grabbed', 'the', 'milk', 'there', '<end>'],
 ['Jeff', 'gave', 'the', 'football', 'to', 'Fred', '<end>'],
 ['Fred', 'handed', 'the', 'football', 'to', 'Jeff', '<end>'],
 ['Jeff', 'handed', 'the', 'football', 'to', 'Fred', '<end>'],
 ['Fred', 'gave', 'the', 'football', 'to', 'Jeff', '<end>']]

Question: ['Who', 'did', 'Fred', 'give', 'the', 'football', 'to', '?']

Ground Truth: ['Jeff', '<end>']

- - - - - - - - - - - - 
Machine Answer: ['Jeff', '<end>']
```
