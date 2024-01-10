# Joint Chinese Word Segmentation and Part-of-Speech Tagging with the Transformer
This is the code accompanying my research project paper on Joint Chinese Word Segmentation and Part-of-Speech Tagging with the Transformer.
First, check out requirements.txt

To train, you need to obtain the data yourself first from the Penn Chinese Treebank (CTB7, CTB9). Specific Instructions on data processing can be found under Data_Processing.
Once you obtain the data and followed the instructions under Data_Processing, you can run the following. This will clean all the data in a form we want for training, you will be prompted to 
select a version to train. 
```shell
$ chmod 700 run_train.sh     
$ ./run_train.sh   
```
To evaluate the model, you can run (but you may have to manually change the code **for now**, if you want to evaluate different versions)
```shell
$ python3 evaluate.py
```
### Example run
If you just want to check out the model, I can share a trained version of CTB9, then you can run
```shell
$ python3 priv_test.py
```
It looks something like this:
![Example Run](example.png "Example Run")
### Training loss for the first 500 steps
![Training Loss](fig3.png "Training loss for first 500 steps")
### Embeddings for the Decoder
![Embeddings for Decoder](fig2.jpg "Embeddings for Decoder")