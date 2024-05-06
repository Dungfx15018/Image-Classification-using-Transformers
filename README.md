# Image Classification using Transformer

Thử nghiệm với Colab

<a href="https://colab.research.google.com/drive/1Biycga_f1sUVdzPVR7Fwfo_IogSYez0J?usp=sharing"><img src="https://storage.googleapis.com/protonx-cloud-storage/colab_favicon_256px.png" width=80> </a>

Architecture Image
<p align="center">
    <img src='https://storage.googleapis.com/protonx-cloud-storage/images/arc.PNG' class="center">
</p>





##  Set up environment
- Step 1: 

```python
create requirements.txt
```

- Step 2: 
```
pip install -r requirements.txt
``` 

## II.  Set up your dataset

- Guide user how to download your data and set the data pipeline <--- **FIXME**
- References: [NLP](https://github.com/bangoc123/transformer) and [CV](https://github.com/bangoc123/mlp-mixer)

## III. Training Process


Training script:




```python
!python data.py

!python train.py --data-dir $data_dir --checkpoint $checkpoint --test-size 0.2 --gradient-accumulation-steps 4 --learning-rate 5e-5 --per-device-train-batch-size 16 --per-device-eval-batch-size 16 --max-steps 2000 --num-train-epochs 20 --adam-epsilon 1e-8 --adam-beta1 0.9  --adam-beta2 0.99  --logging-steps 20 --warmup-ratio 0.1 
``` 


There are some important arguments for the script you should consider when running it:
- `test-size`: 
- `data-dir`: dataset
- `checkpoint`: model pretrained
- `gradient-accumulation-steps`:
- `learning-rate`:
- `per-device-train-batch-size`:
- `per-device-eval-batch-size`:
- `max-steps`:
- `num-train-epochs`:
- `adam-epsilon`:
- `adam-beta1`:
- `adam-beta2`:
- `logging-steps`:
- `warmup-ratio`:


## Result 



Your implementation
```
Epoch	Training Loss	Validation Loss	            Accuracy
1	        5.012900	    4.945810	            0.042895
2	        4.699800	    4.651641	            0.403932
3	        4.278800	    4.195499	            0.702412
4	        3.819400	    3.827306	            0.765862
5	        3.496500	    3.492244	            0.804289
6	        3.133700	    3.213509	            0.831099
7	        2.878900	    2.985585	            0.830205
8	        2.637500	    2.750561	            0.848078
9	        2.452400	    2.569777	            0.854334
10	        2.239300	    2.395381	            0.878462
11	        2.063700	    2.265703	            0.885612
12	        1.943000	    2.162457	            0.873994
13	        1.848700	    2.085841	            0.876675
14	        1.765900	    1.983207	            0.890974
15	        1.709900	    1.948199	            0.892761
16	        1.667100	    1.910196	            0.895442
17	        1.623400	    1.896973	            0.894548

```


## Testing model with new image
```python
!python predict.py --dir-model 'pokemon_models/checkpoint-1610' --image-path '0cfe57a5bf674650b0de0c381df13ca0_jpg.rf.cf29339aa61d57131478f066ba7cceba.jpg'  --checkpoint  'google/vit-base-patch16-224-in21k' --data-dir 'fcakyon/pokemon-classification'  --test-size 0.2 


[{'score': 0.7122378349304199, 'label': 'Alakazam'}, {'score': 0.03247630596160889, 'label': 'Kadabra'}, {'score': 0.0074781812727451324, 'label': 'Abra'}, {'score': 0.0038069335278123617, 'label': 'Farfetchd'}, {'score': 0.0037275038193911314, 'label': 'Beedrill'}]
```
<a href="https://huggingface.co/ILT37/Image-classification"><img src="https://storage.cloud.google.com/favicon_huggingface/6308b81d61b3e2a522f01466.png" width=80> </a>


## Authors:
- Github: https://github.com/Dungfx15018
- Email: dungtrandinh513@gmail.com
- Github: https://github.com/EveTLynn
- Email: linhtong1201@gmail.com

## Advisors:
- Github: https://github.com/bangoc123
- Email: protonxai@gmail.com

