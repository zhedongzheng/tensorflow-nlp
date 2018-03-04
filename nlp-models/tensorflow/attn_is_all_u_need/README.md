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

    * enhanced masking

    * decoding on graph

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
    INFO:tensorflow:Loss for final step: 4.581911.
    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ./saved/model.ckpt-7092
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    你是谁 -> 我是小通
    你喜欢我吗 -> 我喜欢你
    给我唱一首歌 -> =。=========
    我帅吗 -> 你是我的
    ```

<img src="https://github.com/zhedongzheng/finch/blob/master/assets/transform20fps.gif" height='400'>
