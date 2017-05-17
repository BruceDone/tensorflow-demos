# tensorflow-demos
all kinds of demos of tensorflow code 

# environment
* python: 2.7.12
* system: ubuntu 14.04
* tensorflow: tensorflow-gpu==1.0.1 (if you do not has the gpu ,just use the cpu version)

# demo list 
* [x] crack_captcha -- gen image to crack the captcha.
* [x] crack_captcha_rgb -- since the captcha will in rgb mode ,so just train a model to crack it.
* [x] crack_captcha_3d -- the data source comes from the sogou wexin capthca.
* [ ] mnist
* [ ] cat_vs_dog 



## crack_captcha
* install the requirements 
* `python gen_model.py` it will start to train the model 
* after finish train model ,`python validate.py` will do validate the image result

## crack_captcha_rgb
* install the requirements
* `python gen_model.py` it will start to train the model
* after finish train model ,`python validate.py` will do validate the image result

## crack_captcha_3d

* install the requirements
* `python train_model.py` it will start to train the model
* after finish train model ,`python validate.py` will do validate the image result

