# Bach unfinished fugue - LSTM music generation
**This project was one of the project suggestions for a course at my university (University of Groningen). I worked on the project with my teammates, but did not get (for me) satisfactory results, partly due to time constraints. We originally used a simple LSTM model (single small layer, implemented in tensorflow). I wanted to do it differently and started from scratch using PyTorch. All code in this repository folder is written by me. The description of the project is as follows:**
 
Long time ago (1993) there was a time series prediction competition organized by the Santa Fe
Institute which became rather famous in the community. The 6 data sets used for that competition have entered
the folklore of the time series prediction community. Among the data sets, the last – and by far most difficult one
– is an unfinished fugue written by Bach. The task was to use machine learning to complete the composition of
this fugue. Achieving that would amount to finishing a genius' work… In retrospect, one can marvel at the
audacity and innocence of the competition organizers; this task is of an unfathomable difficulty and far from
being solved even today. But it is fun to try one's best and see what piece of artificial music art one can get with
the machine learning tools that one masters. 

The unfinished fugue in question: ***Contrapunctus XIV: https://www.youtube.com/watch?v=JbM3VTIvOBk***

An original source paper: ***santa_fe_competition.pdf***

## Data:
Given is a text file (input.txt) which consist of 4 sequences of integer numbers representing the 4 different voices of the fugue (voices = individual parts of the music piece played simultaniously). The integer numbers represent the pitch of the voice at that current point in time. When the pitch stays the same for multiple steps in time for a single voice, the pitch is supposed to be played for the entire duration. Every timestep is 1/16th of a bar.

example (step 1251-1266):

61	55	52	47 <br>
61	55	52	47 <br>
61	55	49	47 <br>
61	55	49	47 <br>
61	55	49	47 <br>
61	55	49	47 <br>
73	54	54	46 <br>
73	54	54	46 <br>
73	54	54	46 <br>
73	54	54	46 <br>
73	54	52	46 <br>
73	54	52	46 <br>
73	54	52	46 <br>
73	54	52	46 <br>
66	54	52	47 <br>
66	54	52	47 <br>

## Model architecture (```LSTM_bach.ipynb / LSTM_bach.py```)
The model consist of a convolutional LSTM with 3 convolutional layers followed by a double LSTM layer.

### Hyperparameter testing
The following hyperparameters were all tested (as this was my introduction to LSTMs):
- window size: [16, 24, 32, 64]
- convolutional channels (gets doubled for each consecutive conv layer): [8, 16]
- lstm hidden units: [8, 24, 64, 128]
- number of stacked lstm layers: [1, 2] (after other params)

These configurations were run for 100 epochs to determine train/test trajectory.

## Evaluation
The train test split was configure as follows: the first 90 percent of the dataset was used as training data, while the last (which follows the train data) 10 percent was used as test data.  
The data was standardized, making sure no information leaks occured. The scaler was fit on the train data and used to scale both the test and train data. Additionally, the sliding window input data which leaked time information from the train set into the test set were removed.

We found the dataset to be too small, with the test data being not representative of the train data (features learned in train data did not generalize well to test data), resulting in models extracted using early stopping not generating good results.

The lowest test losses occured quickly after the start of training, meaning the models overfit quickly. Even when using really small models (~8/16 hidden units, 1/2 conv layers with 8 channels, subsential dropouts and regularization) the models quickly overfit. Larger models sometimes achieved even lower test loss, but overfit thereafter quickly. It seemed that the current configuration of test/train split was not working as well as I would have liked. 

The next step I took was making a model with a large sliding window of 80 notes, with a large double LSTM layer of 256 units. My thought process was, as I couldn't rely as well on the test loss as I would've liked, I would train a substentially sized network and keep the complexity in check using dropout and weight decay/l2 regularization. The dropout was tuned by running different dropout frequencies and subjectively judging the generated music. I trained the model using google colab to a small training loss, and used this to generate bach-like music, which for the first time resulted in generated music that sounded like actual music. The outputs of different tests can be found in ***/output/...***.

## Conclusion
The dataset size and current train/test split configuration are not sufficient for good test evaluation.

The first next step I thought of in this project would be try a different train/test config. My first idea is to take samples from different time point in the dataset instead of all at the end as the last part of the fugue is too different from the rest. A problem with this however, is that we need to prevent time leaks, and therefore, can't use the all the samples which have some of the notes of all those time windows taken for the test set. For example, if we have a sliding window of 80 time steps, which we use to predict the time step after that, we can't use any of the previous 79 sliding windows of a sliding window (sample) in the test set. All these 80 windows, which can be used to predict any of the time steps in this 80 time step window, will have to be removed. As the dataset is already small, this does not seem to be the best idea. 

Therefore, the logical next step making a well generalizing bach music generation model would be to find a different bigger dataset, which could be used to really learn patterns in the music that translate well to a test set. My solution to this was to tune dropout and regularization untill the model was able to generate music that continued the fugue which actually sounded like music, and most importantly, a possible ending to the fugue itself. In this, I paid extra attention to the first part of the generated music (continuing from where bach ended) as this is where overfit models struggled the most. The most overfit models (low dropout, low regularization) would catch itself in its mistakes later and produce decent results after sounding really bad, probably because they accidentally fall into some pattern that is in the training data and continue from here. If the model continues well where bach left off, I was content with how well it generalized (high dropout/regu models sounded worse after first part than overfit).

The goal of this project was to finish the bach fugue, and eventhough the model is overfit compared to our test set, the generated music sounds pretty good. The model starts off really well, but gets in trouble after a while. However, the parts where you can here the model struggle still sounds greatly better than the models with the lowest test loss. Probably, there are small sections which are very similar to the training music after prediction by the model, but I am not sure if I have a problem with that in this context of the Santa Fe competition. 

Additionally, good musical knowledge + postprocessing/sampling of output of the lstm wouldn't hurt.

