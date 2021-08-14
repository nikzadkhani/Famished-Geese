# Famished-Geese

## Dueling Deep Q-Network
### Directory Structure

```
📦DDQN
 ┣ 📂runs
 ┃ ┣ 📂dueling-net-12
 ┃ ┣ 📂toroidal-net-12
 ┃ ┗ 📂geese-net-12
 ┣ 📂saved_models
 ┃ ┗ 📜*.pt
 ┣ 📜agent.py
 ┣ 📜graphics.py
 ┣ 📜main.py
 ┣ 📜model.py
 ┣ 📜options.py
 ┣ 📜play.ipynb
 ┣ 📜requirements.txt
 ┣ 📜test.py
 ┣ 📜test.sh
 ┣ 📜test_options.json
 ┣ 📜train-dueling.sh
 ┣ 📜train-geesenet.sh
 ┣ 📜train-toroidal.sh
 ┣ 📜train.py
 ┣ 📜train.sh
 ┣ 📜train_options.json
 ┣ 📜train_small.sh
 ┣ 📜utils.py
 ┗ 📜vector.py
 ```
 I recommend creating a virtual-environment to run the 
 scripts in this folder. You can install it using pip.
```
pip3 install virtualenv
```
 This code was run with python version 3.8.6. You can create and activate your virtual environment in the current directory with the following command
 ```
 virtualenv geese
 source geese/bin/activate
 ```
 Ensure that your current directory is within the DDQN folder, then
 you can run
 ```
 pip3 -r install requirements.txt
 ```
 to download all the dependencies. Now you should be able to run
 `train-dueling.sh`, using
 ```
 sh train-dueling.sh
 ```
 To model will be stored in the `saved_models` folder and the tensorboard logs will be in the `runs/dueling-net-12' folder`. To view the different
 options that you can change and input into `main.py` you can check `options.py`. `train.py` contains the training loop which calls the DDQN
 model from `model.py` for training updates and such. `test.py` is used to test the models in `saved_models` and will load up whatever you model you 
 pass into it.

 If you run into any issues where the `kaggle_environments` package is throwing an error stating that `gfootball` is not downloaded then don't fret you just need to delete the `football` environment from the `kaggle`_environment` environment folder. You can find this folder in your python package install location or if you use the virtual environment, you can find it here
 ```
 rm -r geese/lib/python3.8/site-packages/kaggle_environments/envs/football
 ```
 Note you may have a different version of python so your directory structure may be slightly different. The hungry geese environment isn not actually dependent on the football package that is why we can delete it.


 If you want to see the actual actions and observations per time step you can change `render` to `True` in `main.py` and then run `test.sh`.

 Lastly you can run tensorboard using the following command from the DDQN directory.

 ```
 tensorboard --logdir=runs
 ```



## Deep Q-Network
### Objectives
#### Training:
Run Train.py to start training the agent. 
#### Checkpointing:
Train.py creates a variable for a checkpoint directory that can be either restored or initialized. 
#### Logging and visualizations:
You can view existing logs from the commandline by typing:

```
Tensorboard --logdir logs-big
```

This will open up a web page in your browser and allow you to review past tensorboard events as well as stream the results of current training runs.

Link to google slides

https://docs.google.com/presentation/d/1FcyikhtENs2Bqo-xRVE65cZJX57OnCBr/edit?usp=sharing&ouid=113817137264964264363&rtpof=true&sd=true
