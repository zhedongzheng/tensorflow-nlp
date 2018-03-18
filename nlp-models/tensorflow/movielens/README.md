<img src="https://github.com/zhedongzheng/finch/blob/master/assets/movielens.png">

You need Python 2 for this sub-project, because we need to use [PaddlePaddle](http://www.paddlepaddle.org/) for pre-processed Movielens dataset

```
pip install paddlepaddle
```

```
cd ./data
python movielens_paddle.py
cd ..
python train.py
```
```
Epoch [29/30] | Batch [0/3516] | Loss: 2.60
Epoch [29/30] | Batch [500/3516] | Loss: 1.85
Epoch [29/30] | Batch [1000/3516] | Loss: 1.54
Epoch [29/30] | Batch [1500/3516] | Loss: 2.51
Epoch [29/30] | Batch [2000/3516] | Loss: 2.21
Epoch [29/30] | Batch [2500/3516] | Loss: 5.93
Epoch [29/30] | Batch [3000/3516] | Loss: 3.27
Epoch [29/30] | Batch [3500/3516] | Loss: 2.10
------------------------------
Testing losses: 2.97667910193
Prediction: 2.69, Actual: 3.00
Prediction: 1.94, Actual: 3.00
Prediction: 2.00, Actual: 3.00
Prediction: 3.32, Actual: 3.00
Prediction: 3.39, Actual: 3.00
------------
```
