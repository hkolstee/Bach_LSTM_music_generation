# Bach unfinished fugue
### This project was one of the project suggestions for a course at my university (University of Groningen). I worked on the project with my teammates, but did not get (for me) satisfactory results. We originally used a simple LSTM model (single layer, implemented in tensorflow). I wanted to do it differently and started from scratch using PyTorch. The description of the project is as follows:
 
Long time ago (1993) there was a time series prediction competition organized by the Santa Fe
Institute which became rather famous in the community. The 6 data sets used for that competition have entered
the folklore of the time series prediction community. Among the data sets, the last – and by far most difficult one
– is an unfinished fugue written by Bach. The task was to use machine learning to complete the composition of
this fugue. Achieving that would amount to finishing a genius' work… In retrospect, one can marvel at the
audacity and innocence of the competition organizers; this task is of an unfathomable difficulty and far from
being solved even today. But it is fun to try one's best and see what piece of artificial music art one can get with
the machine learning tools that one masters. 

The unfinished fugue in question: ***Contrapunctus XIV: https://www.youtube.com/watch?v=JbM3VTIvOBk***

### Data:
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
