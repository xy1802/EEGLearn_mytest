# EEGLearn_mytest
Try to show the pictures in EEGLearn, add model save and load to do prediction. I use Pytorch in this project.  
The original method is from https://github.com/VDelv/EEGLearn-Pytorch  
I just changed it a little bit, which may help you understand the method better.

##Method show
![alt text](diagram.png "Converting EEG recordings to movie snippets")
|:--:| 
| Taken from [Bashivan et al. 2016](https://arxiv.org/pdf/1511.06448.pdf)|

![images](https://github.com/xy1802/EEGLearn_mytest/blob/master/images.png)
|:--:| 
| images in the dataset |

##Some Tips
1. I do some changes in train,utils, and add a file model_load_pre. Most useful tools are in model_load_pre. 
2. The original author train the network one by one, one patient, one group of training data, but I train the network for all the people. 
Because when I try to use patient 15's data to predict patient 1's label, the result is terrible. If you want to train a model with stronger generalization ability,
just use more data to train and more patient data will give you a better result.
3. I use gpu to accelerate my network, if you don't have NVIDA GPU, please cancel all the .gpu() in your code.

## References 

If you are using this code please [cite](Cite.bib) the paper:

Bashivan, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

https://arxiv.org/pdf/1511.06448.pdf
