import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
from fractions import Fraction

logger = logging.getLogger(__name__)

class ZeppelinEnv(gym.Env):
	"""
	Agent is navigating a Zeppelin flying in the wind.
	The wind is composed of a wind field and a sudden turbulence.
	In particular, the agent is navigating near a circular obstacle which the agent must avoid.
	The goal of the agent is to leave the obstacle region.
	"""

	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 60
	}

	def __init__(self):
		# Makes the continuous fragment of the system determinitic by fixing the
		# amount of time that the ODE evolves.
		self.TIME_STEP = 0.5

		self.MAX_WIND_SPEED = 60. # m/s in ~ 200 km/h
		self.MAX_TURBULENCE = 20. # m/s in ~ 70 km/h
		self.MAX_VELOCITY = 40. # m/s in ~ 140 km/h

		self.INCLUDE_UNWINNABLE=True

		self.MAX_X = 5000. # m
		self.MIN_X = -500. # m
		self.MAX_Y = 5000. # m
		self.MIN_Y = -5000. # m

		self.WORST_CASE_TURBULENCE=False

		self.MIN_C = 50. # m
		self.MAX_C = 3000. # m

		self.RENDER_ZEPPELIN_RADIUS = 100
	
		# Action Space:
		#   - y1
		#   - y2
		#   Must be the case that (y1**2 + y2**2)**0.5 <= 1
		action_low = np.array([-1.0, -1.0])
		action_high = np.array([1.0, 1.0])
		self.action_space = spaces.Box(action_low, action_high)

		# Observation Space:
		#   0 - Position x1 \in [MIN_X, MAX_X]
		#   1 - Position x2 \in [MIN_Y, MAX_Y]
		#   2 - Position o1 \in [MIN_X, MAX_X]
		#   3 - Position o2 \in [MIN_Y, MAX_Y]
		#   4 - Wind v1 \in [-MAX_WIND_SPEED,MAX_WIND_SPEED]
		#   5 - Wind v2 \in [-MAX_WIND_SPEED,MAX_WIND_SPEED]
		#   6 - Obstacle radius c \in [MIN_C, MAX_C]
		#   Must be the case that (v1**2 + v2**2)**0.5 <= MAX_WIND_SPEED
		obs_low = np.array([
			self.MIN_X,
			self.MIN_Y,
			self.MIN_X,
			self.MIN_Y,
			-self.MAX_WIND_SPEED,
			-self.MAX_WIND_SPEED,
			self.MIN_C,
		])
		obs_high = np.array([
			self.MAX_X,
			self.MAX_Y,
			self.MAX_X,
			self.MAX_Y,
			self.MAX_WIND_SPEED,
			self.MAX_WIND_SPEED,
			self.MAX_C
		])
		self.observation_space = spaces.Box(obs_low, obs_high)

		self._seed()
		self.viewer = None
		self.state = None
	
	def norm(self, e1, e2):
		return (e1**2 + e2**2)**0.5

	def is_crash(self, some_state=None):
		if some_state is None:
			some_state = self.state
		x1 = some_state[0]
		x2 = some_state[1]
		o1 = some_state[2]
		o2 = some_state[3]
		c = some_state[6]
		return self.norm(x1-o1, x2-o2) <= c
	
	def reached_goal(self, some_state):
		x1 = some_state[0]
		x2 = some_state[1]
		# Check if outside of min/max region:
		return x1 > self.MAX_X or x1 < self.MIN_X or x2 > self.MAX_Y or x2 < self.MIN_Y
	
	def x1_min(self, state):
		x2 = state[1]
		c = state[6]
		w = self.norm(state[4], state[5])
		return -(self.TIME_STEP * (self.MAX_VELOCITY + self.MAX_TURBULENCE) + ((self.MAX_VELOCITY - self.MAX_TURBULENCE) / w * (c - (x2 - self.TIME_STEP * (self.MAX_TURBULENCE + self.MAX_VELOCITY+w))) + c) )-200

	def x1_max(self, state):
		x2 = state[1]
		c = state[6]
		w = self.norm(state[4], state[5])
		return self.TIME_STEP * (self.MAX_VELOCITY + self.MAX_TURBULENCE) + ((self.MAX_VELOCITY - self.MAX_TURBULENCE) / w * (c - (x2 - self.TIME_STEP * (self.MAX_TURBULENCE + self.MAX_VELOCITY+w))) + c)+200
	
	def x2_max(self, state):
		c = state[6]
		w = self.norm(state[4], state[5])
		return c + w / (self.MAX_VELOCITY - self.MAX_TURBULENCE) * c + self.TIME_STEP * (self.MAX_VELOCITY + self.MAX_TURBULENCE + w)+200
	
	def x2_min(self, state):
		c = state[6]
		return -c - self.TIME_STEP * (self.MAX_VELOCITY + self.MAX_TURBULENCE)-200



	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		return self._stepByModel(action)

	def _stepByModel(self, action):
		assert self.action_space.contains(action), "%s (of type %s) invalid" % (str(action), type(action))
		state = self.state

		# Compute turbulence
		z1 = Fraction(0.0)
		z2 = Fraction(-1.0)
		if self.WORST_CASE_TURBULENCE:
			z1, z2 = self.get_worst_turbulence(state)
		else:
			z1_norm = self.np_random.uniform(low=-1.0, high=1.0, size=(1,))[0]
			z2_norm = np.sqrt(1-z1_norm**2)
			turbulence_strength = self.np_random.triangular(-self.MAX_TURBULENCE, 0.0 ,self.MAX_TURBULENCE, size=(1,))[0]
			z1 = Fraction(z1_norm * turbulence_strength)
			z2 = Fraction(z2_norm * turbulence_strength)

		x1 = state[0]
		x2 = state[1]
		o1 = state[2]
		o2 = state[3]
		v1 = state[4]
		v2 = state[5]
		c = state[6]

		t = self.TIME_STEP
		
		y1 = Fraction(self.MAX_VELOCITY)*Fraction(float(action[0]))
		y2 = Fraction(self.MAX_VELOCITY)*Fraction(float(action[1]))

		x1_new = x1 + t*( y1 + z1 + v1 )
		x2_new = x2 + t*( y2 + z2 + v2 )
		
		state = (x1_new, x2_new, o1, o2, v1, v2, c)

		has_crashed = self.is_crash(state)
		reached_goal = self.reached_goal(state)
		done = has_crashed or reached_goal
		done = bool(done)

		# Imaginary fuel -> try to work as fast as possible

		if has_crashed:
			# Penalize for crashing
			reward = -100000
		elif reached_goal:
			reward = 1
		else:
			reward = 0
		self.state = state
		obs_state = list(map(float, state))

		return np.array(obs_state), reward, done, {'crash': has_crashed, 'goal': reached_goal}
	
	def is_in_bounds(self, state):
		w = state[3]
		c = state[2]
		x1 = state[0]
		x2 = state[1]
		intermediate_state1 = (None, None, c, w)
		if x2 < self.x2_min(intermediate_state1) or x2 > self.x2_max(intermediate_state1):
			#print("o", end="")
			return False
		intermediate_state2 = (None, x2, c, w)
		if x1 < self.x1_min(intermediate_state2) or x1 > self.x1_max(intermediate_state2):
			#print("o", end="")
			return False
		return True

	def random_reset(self):
		epsilon = 0.1

		v1_norm = self.np_random.uniform(low=-1.0, high=1.0, size=(1,))[0]
		v2_norm = np.sqrt(1-v1_norm**2)
		v_strength = self.np_random.uniform(self.MAX_VELOCITY - self.MAX_TURBULENCE ,self.MAX_WIND_SPEED)
		v_dir = 1. if random.random() < 0.5 else -1.
		v_strength = v_dir * v_strength
		v1 = Fraction(v1_norm * v_strength)
		v2 = Fraction(v2_norm * v_strength)

		c = self.np_random.uniform(low=self.MIN_C, high=self.MAX_C, size=(1,))[0]
		o1 = self.np_random.uniform(low=self.MIN_X, high=self.MAX_X, size=(1,))[0]
		o2 = self.np_random.uniform(low=self.MIN_Y, high=self.MAX_Y, size=(1,))[0]
		intermediate_state1 = (None, None, o1, o2, v1, v2, c)
		x2 = self.np_random.uniform(low=self.x2_min(intermediate_state1), high=self.x2_max(intermediate_state1), size=(1,))[0]
		intermediate_state2 = (None, x2, o1, o2, v1, v2, c)
		x1 = self.np_random.uniform(low=self.x1_min(intermediate_state2), high=self.x1_max(intermediate_state2), size=(1,))[0]

		self.state = list(map(Fraction,map(float,(x1, x2, o1, o2, v1, v2, c))))
			
		return np.array(self.state)

	def exclude_because_unwinnable(self, state,print_vals=False):
		"""
		Returns True if state should be included, because setup is unwinnable (i.e. inside Bermuda triangle)
		"""
		if self.INCLUDE_UNWINNABLE:
			return False
		x1 = state[0]
		x2 = state[1]
		o1 = state[2]
		o2 = state[3]
		v1 = state[4]
		v2 = state[5]
		c = state[6]
		vnorm = self.norm(v1, v2)
		a1 = x1 - o1
		a2 = x2 - o2
		q1 = (-c/(self.MAX_VELOCITY - self.MAX_TURBULENCE)*v1)
		q2 = (-c/(self.MAX_VELOCITY - self.MAX_TURBULENCE)*v2)
		qSquared = (q1**2 + q2**2)
		factor = (q1**2 + q2**2 - c**2)**0.5
		if print_vals:
			print(float(a1*(v1/vnorm) + a2*(v2/vnorm) - c))
			print(float(c*(q1*(a1 - q1) + q2*(a2 - q2)) + factor*(q1*a2 - q2*a1)))
			print(c*(q1*(a1 - q1) + q2*(a2 - q2)) - factor*(q1*a2 - q2*a1))
		if a1*(v1/vnorm) + a2*(v2/vnorm) - c > 0:
			# x is in slipstream of obstacle
			return False
		elif c*(q1*(a1 - q1) + q2*(a2 - q2)) + factor*(q1*a2 - q2*a1) > 0.0 or c*(q1*(a1 - q1) + q2*(a2 - q2)) - factor*(q1*a2 - q2*a1) > 0.0:
			# x is on side 1 or 2 of Bermuda triangle
			return False
		else:
			return True
	
	def get_worst_turbulence(self, state):
		x1 = state[0]
		x2 = state[1]
		c = state[2]
		w = state[3]
		x2_min = -c
		x2_max = (c + w / Fraction(self.MAX_VELOCITY - self.MAX_TURBULENCE) * c)
		x1_min = (- (Fraction(self.MAX_VELOCITY - self.MAX_TURBULENCE) / w * (c - x2) + c))
		#x1_max = ( ((self.MAX_VELOCITY - self.MAX_TURBULENCE) / w * (c - x2) + c))
		gamma = Fraction(self.MAX_TURBULENCE)/(w**2+Fraction(self.MAX_VELOCITY - self.MAX_TURBULENCE)**2)**0.5
		if x2 <= x2_min:
			return Fraction(0.), Fraction(self.MAX_TURBULENCE)
		elif x2 >= x2_max:
			return Fraction(0.), Fraction(-self.MAX_TURBULENCE)
		elif x1 <= x1_min:
			return Fraction(gamma*w), Fraction(-gamma*(self.MAX_VELOCITY - self.MAX_TURBULENCE))
		else: # Assume x1 >= x1_max:
			return Fraction(-gamma*w), Fraction(-gamma*(self.MAX_VELOCITY - self.MAX_TURBULENCE))
	
	def model_reset(self):
		#print("m")
		while True:
			res = self.random_reset()
			if not self.is_crash(res) and not self.reached_goal(res) and not self.exclude_because_unwinnable(res):
				rv = res
				break
		return rv
	
	def reset(self):
		return self.model_reset()

	def render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		screen_width = 800
		screen_height = 800

		world_size_x = self.MAX_X - self.MIN_X
		world_size_y = self.MAX_Y - self.MIN_Y
		world_offset_x = -self.MIN_X
		world_offset_y = -self.MIN_Y
		scale_x = screen_width/world_size_x
		scale_y = screen_height/world_size_y
		from gym.envs.classic_control import rendering
		if self.viewer is None:
			self.viewer = rendering.Viewer(screen_width, screen_height)

			# Obstacle Circle
			obstacle = rendering.make_circle(radius=1.0, filled=True)
			obstacle.set_color(1.0, 0.0, 0.0)
			self.obstacletrans = rendering.Transform()
			obstacle.add_attr(self.obstacletrans)
			self.viewer.add_geom(obstacle)
			self.obstacletrans.set_translation(world_offset_x*scale_x, world_offset_y*scale_y)

			# Zeppelin
			zeppelin = rendering.make_circle(self.RENDER_ZEPPELIN_RADIUS*scale_x)
			zeppelin.set_color(0.0, 1.0, 1.0)
			self.zeppelintrans = rendering.Transform()
			zeppelin.add_attr(self.zeppelintrans)
			self.viewer.add_geom(zeppelin)

		if self.state is None: return None
		c=self.state[6]
		w = self.norm(self.state[4], self.state[5])

		# Set Obstacle Size
		self.obstacletrans.set_scale(c*scale_x,c*scale_y)
		o1 = float(self.state[2]+world_offset_x)*scale_x
		o2 = float(self.state[3]+world_offset_y)*scale_y
		self.obstacletrans.set_translation(o1,o2)

		# Set Zeppelin Position:
		x1 = float(self.state[0]+world_offset_x) * scale_x
		x2 = float(self.state[1]+world_offset_y) * scale_y


		self.zeppelintrans.set_translation(x1, x2)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

gym.register(
	  id='realZeppelin-v0',
	  entry_point=ZeppelinEnv,
	  max_episode_steps=1200,  # todo edit
	  reward_threshold=400.0, # todo edit
  )