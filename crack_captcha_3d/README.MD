# Folder 
* test 
* train
* validate 

test folder for test data ,train for train data for model , when you finish train data ,you can use the validate folder to validate your image 

## How to use 
* download the captcha image from the url of :http://mp.weixin.qq.com/mp/verifycode?cert=1492745324932.2913 -- it has changed rule now ,please pay attention
* label the image as demo of data folder image 
* if you want the high accuracy ,please label as much as you can ,it will really increase the accuracy
* update the `config.py` relay on your situation
* just run the `train_model.py` script ,it will start to train