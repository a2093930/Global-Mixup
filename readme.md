# Global Mixup: Global relationships Based Data Augmentation for Text Classifification



## Data：

You need to convert the file to csv format, and organizes the data in the following form:



**label,sentence**



for example:

2,Offers a breath of the fresh air of true sophistication .



## Run



#### **trian the model:**

**for CNN and LSTM：**

python run.py --dir  DIR of train,val,test \

--input_train train_name \

--input_val val_name \

--input_test test_name \

--num_mix T Parameter \

--alpha 4  Beta Parameter \alpha \

--classes 2 Number of target labels \

--begin begin of target labels





for example:

python run.py --dir sst-2 --input_train sst-2_train_500.csv --input_val sst-2_val.csv --input_test sst-2_test.csv --num_mix 8 --alpha 4 --classes 2 --begin 1



**for BERT:**

You can change the specific parameters in the classifier_single.py  or classifier.py

python classifier_single.py #test single data set for 10 times

python classifier.py #test all data set for 10 times
