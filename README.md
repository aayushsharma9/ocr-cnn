# OCR - CNN
### Overview
Machine learning model trained on a Convolutional Neural Network to recognize handwritten digits and characters of English Alphabet.
The project uses the [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19) which contains a labelled image dataset of 62 classes i.e., 10 digits and 52 uppercase and lowercase characters from the English Alphabet.
The data is then further balanced for each class and then augmented to generate a larger set of images. Also, similar classes for similar looking alphabets such as O and o, L and l, i and I etc. are merged into one which makes a total of 47 classes.

### Model Summary

**Layer type**              |         **Output Shape**	        |       **Params**  |
----------------------------|-----------------------------------|-------------------|
conv2d (Conv2D)             |       (None, 30, 30, 32)          |       320         |
batch_normalization         |       (None, 30, 30,32)           |       320         |
conv2d_1 (Conv2D)           |       (None, 24, 24, 32)          |       9248        |
batch_normalization_1       |       (None, 28, 28, 32)          |       128         |
conv2d_2 (Conv2D)           |       (None, 14, 14, 32)          |     	25632       |
batch_normalization_2       |       (None, 14, 14, 32)	        |       128         |
dropout (Dropout)           |       (None, 14, 14, 32)          |  	    0           |
conv2d_3 (Conv2D)           |       (None, 12, 12, 64) 	        |       18496       |
batch_normalization_3       |       (None, 12, 12, 64) 	        |       256         |
conv2d_4 (Conv2D)           |       (None, 10, 10, 64)	        |       36928       |
batch_normalization_4       |       (None, 10, 10, 64)  	    	|       256         |
conv2d_5 (Conv2D)           |       (None, 5, 5, 64) 	        	|       102464      |
batch_normalization_5       |       (None, 5, 5, 64) 	        	|       256         |
dropout_1 (Dropout)         |       (None, 5, 5, 64) 	        	|       0           |
conv2d_6 (Conv2D)           |       (None, 2, 2, 128) 	        |       131200      |
batch_normalization_6       |       (None, 2, 2, 128) 	        |       512         |
flatten (Flatten)           |       (None, 512)  	            	|       0           |
dropout_2 (Dropout)         |       (None, 512) 	            	|       0           |
dense (Dense)               |       (None, 48)   	            	|       24624       |

Total params: 4,479,344<br/>
Trainable params: 4,478,512<br/>
Non-trainable params: 832<br/>

### Tensorboard Results

![Legend](https://i.imgur.com/x1Lq4uA.png)<br/>
Legend
![Sparse Categorical Accuracy Comparison per Epoch](https://i.imgur.com/WLQMfPF.png)
<p align=center>Sparse Categorical Accuracy Comparison per Epoch</p>
