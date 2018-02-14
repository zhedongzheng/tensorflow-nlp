---
Implementing the idea of ["Attention is All you Need"](https://arxiv.org/abs/1706.03762)

---

<img src="https://github.com/zhedongzheng/finch/blob/master/assets/transformer.png" width="300">

Some functions are adapted from [Kyubyong's](https://github.com/Kyubyong/transformer) work, thanks for him!

* Based on that, we have:
    * implemented the model under the architecture of ```tf.estimator.Estimator``` API

    * added an option to share the weights between encoder embedding and decoder embedding

    * added an option to share the weights between decoder embedding and output projection

    * added the learning rate variation according to the formula in paper, and also expotential decay

    * added more activation choices (leaky relu / elu) for easier gradient propagation

    * enhanced masking (mask positional encoding as well)

    * implemented decoding on graph and added start tokens

* Small Task 1: learn sorting characters

    ```  python train_letters.py --tied_embedding --label_smoothing ```
        
    ```
    INFO:tensorflow:lr = 0.000914113 (22.901 sec)
    INFO:tensorflow:Saving checkpoints for 4000 into ./saved/model.ckpt.
    INFO:tensorflow:Loss for final step: 0.707183.
    INFO:tensorflow:Restoring parameters from ./saved/model.ckpt-4000
    apple -> aelpp
    common -> cmmnoo
    zhedong -> deghnoz
    ```

* Small Task 2: learn chinese dialog

    ``` python train_dialog.py```
    
    ```
    INFO:tensorflow:loss = 3.48946, step = 7001 (19.987 sec)
    INFO:tensorflow:lr = 0.000851138 (19.987 sec)
    INFO:tensorflow:Saving checkpoints for 7092 into ./saved/model.ckpt.
    INFO:tensorflow:Loss for final step: 3.32809.
    INFO:tensorflow:Restoring parameters from ./saved/model.ckpt-7092
    你是谁 -> 我是小通
    你喜欢我吗 -> 我喜欢你
    给我唱一首歌 -> = =========
    我帅吗 -> = =========
    ```

<img src="https://github.com/zhedongzheng/finch/blob/master/assets/transform20fps.gif" height='400'>
