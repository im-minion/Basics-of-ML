import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

# learning rate
LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
intial_games = 10000

def some_random_games_first():
	for episode in range(5):
		env.reset()
		for t in range(goal_steps):
			env.render()
			# if you want faster then dont render
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			# obsvartion -> array of data pixel data (positions)
			# reward 0 or 1 bounced or not
			# done -> game over or not
			# info -> other information
			if done:
				break

some_random_games_first()