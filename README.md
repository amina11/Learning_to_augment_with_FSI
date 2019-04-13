# README #

### This repository is for the learning to augment simplified model (ICDM_2018)  

 It includes code for artificial dataset and real world dataset experiment   

### How do I get set up?   

*You will need python 2.7  
*Tensorflow 1.7.0  
*Tensorboard  
*mlxtend  


### Files description  
* In both artificial and real world data folders, there is a file called main.py, that is the main file which will read the data, call the functions in uitilities to build the graph, and train the model  
* The file called cross validation.py is used for parameter tuning, it can be called by the call.sh bash file.  
* The artificial data folder also include artifical data generation code. This code is saved in ipynb for more interactive convinience.
* Most used functions and arguments are defined in utility.py file. It will be imported later on in main.py
* utility.py also include McNemar test function, which can be used for result analysis.  
* Realworld data file also includes the code for gaussian noise, after the best sigma is found by using cross validation, main.py can be called by chanding the final train data augmentation to the gaussian data augentation defined in utilities file. 