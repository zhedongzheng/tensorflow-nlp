<img src="https://github.com/zhedongzheng/finch/blob/master/assets/end2end_mn.png">

```
python train.py
```
```
[10/10] | [150/156] | loss:0.262 | acc:0.883
final testing accuracy: 0.928

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
