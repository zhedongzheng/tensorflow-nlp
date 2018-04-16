<img src="https://github.com/zhedongzheng/finch/blob/master/assets/dmn-details.png">

* Many functions are adapted from [Alex Barron's](https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow) work, thanks for him!
* Based on that:
    * We have used ```tf.map_fn``` to replace the Python for loop, which makes the model independent of sequence length
    * We have added a decoder in the answer module for "talking"
    * We have reproduced ```AttentionGRUCell``` from official ```GRUCell``` from TF 1.4

```
python train.py
```
```
Data Shuffled
[10/10] | [0/156] | loss:0.028 | acc:0.984
[10/10] | [50/156] | loss:0.002 | acc:1.000
[10/10] | [100/156] | loss:0.020 | acc:0.984
[10/10] | [150/156] | loss:0.004 | acc:1.000
             precision    recall  f1-score   support

          4       1.00      1.00      1.00        98
         10       0.99      0.99      0.99       129
         28       0.99      0.99      0.99       164
         32       0.99      1.00      1.00       188
         34       1.00      0.99      0.99        95
         35       0.99      1.00      1.00       137
         40       1.00      1.00      1.00       189

avg / total       1.00      1.00      1.00      1000


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
