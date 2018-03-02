<img src="https://github.com/zhedongzheng/finch/blob/master/assets/end2end_mn.png">

```
python train.py
```
```
Data Shuffled
[10/10] | [0/156] | loss:0.170 | acc:0.938
[10/10] | [50/156] | loss:0.160 | acc:0.938
[10/10] | [100/156] | loss:0.161 | acc:0.953
[10/10] | [150/156] | loss:0.102 | acc:0.961
final testing accuracy: 0.945

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
