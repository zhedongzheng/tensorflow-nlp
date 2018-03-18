<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_7.png" height='150'>

Please additionally install these packages
```
pip3 install bs4 python_speech_features
```
Download data
```
python data_downloader.py
```
You should see outputs like this:
```
28 /bitstream/1807/24488/28/OAF_thumb_neutral.wav
29 /bitstream/1807/24488/29/OAF_thought_neutral.wav
30 /bitstream/1807/24488/30/OAF_third_neutral.wav
31 /bitstream/1807/24488/31/OAF_thin_neutral.wav
32 /bitstream/1807/24488/32/OAF_tell_neutral.wav
33 /bitstream/1807/24488/33/OAF_team_neutral.wav
```
Train model
```
python train.py
```
You should see following outputs:
```
Prediction: say the word 
Actual: say the word youth
```
