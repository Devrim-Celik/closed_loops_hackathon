# AI 4 Closed-Loop Control Systems - ZF

Group Name: Loopsters Mejores
Group Members: Pascal Schröder, Marc Vidal De Palol, Devrim Celik  
Timeframe: 3 days (10.10.2020 - 12.10.2020)  

## Abstract
This is the github repository of the *Loopsters Mejores*. Our idea revolves generating an "optimal current" which is then used to train an LSTM, effectively avoiding the problem of propagating gradients through the dynamical system. 

## Repositoy
+ ```visualization```: Containg a notebook for visualization.
+ ```gann```: An earlier approach to the problem in which we used genetic algorithms for training a neural network controller. Sadly it took to long with no hope of good performance, but it is fully functional.
+ ```preprocess```: Responsible for decreasing file size of the original dataset and transforming them using different velocities.
+ ```simulation```: Containing our own simulation script using ```odeint``` for speeding up the dynamical systems. Also created an animation.
+ ```sliding_model```: Containing the sliding model, responsible for calculating $I_O$.
+ ```task```: Task related files and information.
