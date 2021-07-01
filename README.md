# Udacity-banana-collector
Udacity p1_navigation project

 
The objective is to navigate around the environment collecting yellow bananas and avoiding blue ones. 
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.


The environment consists of episodic tasks and is considered solved when an average reward of 13 is achieved over 100 episodes.


The states in the environment have 37 dimensions which contains information about the agent’s position, 
velocity and what’s in front of the agent. From each state one of 4 discrete actions can be taken: 
 
	0 - move forward
	1 - move backward
	2 - turn left
	3 - turn right
	



Getting Started

The environment is Windows (64-bit) banana environment: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip

Setting up the environment

1. First we need to create and activate the environment with Python 3.6

		conda create --name drlnd python=3.6 
		activate drlnd

2. Next download the enviroment dependencies with the requirements.txt file

		pip install -r requirements.txt

3. Clone the Github repository 

		git clone https://github.com/alexkelly145/Udacity-banana-collector.git

4. Create an IPython kernel for the drlnd environment

		python -m ipykernel install --user --name drlnd --display-name "drlnd"

5. Make sure drlnd is selected and drlnd appears in the top right corner instead of Python when inside the notebook

Running the code

There are 4 files needed to run the code:

	1. main.ipynb
	2. dqn_agent.py
	3. replay.py
	4. model.py

Make sure all these files are in the same directory.

Open the main.ipynb notebook, the hyperparameters for the DQN agent can be changed in cell 4. 

A number of hypermeters can be changed such as buffer_size, batch_size and whether you want the DQN agent to use prioritized experience replay.

Run all the cells to start training the DQN agent. Once an average score of 13 is achieved the model will save the weights to the current working directory.


