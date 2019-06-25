import gym
import numpy as np
import matplotlib.pyplot as plt

class Planner:

	'''
	Initialization of all necessary variables to generate a policy:
		discretized state space
		control space
		discount factor
		learning rate
		greedy probability (if applicable)
	'''
	def __init__(self,res,gamma,alpha,epsilon,threshold,env):
		self.res = res
		self.Q = np.zeros((np.int((0.5+1.2)/res[0]+1),np.int((0.07+0.07)/res[1]+1),3))
		self.gamma = gamma				#discount factor
		self.alpha = alpha				#learning rate
		self.epsilon = epsilon			#greedy probability
		self.threshold = threshold		#convergence threshold
		self.env = env					#environment
		self.converge = False

	'''
	Learn and return a policy via model-free policy iteration.
	'''
	def __call__(self, policy_type=True):
		return self._td_policy_iter(policy_type)


	#visualize the result
	def visualize(self,Q_1,Q_2,Q_3,policy):
		for i in range(3):
			plt.plot(Q_1[:,i])
			plt.xlabel('iteration')
			plt.ylabel('Q')
			plt.title('Off-policy Q(0,0,%d)'%i)
			plt.show()
		
		for i in range(3):
			plt.plot(Q_2[:,i])
			plt.xlabel('iteration')
			plt.ylabel('Q')
			plt.title('Off-policy Q(-.5,-.01,%d)'%i)
			plt.show()

		for i in range(3):
			plt.plot(Q_3[:,i])
			plt.xlabel('iteration')
			plt.ylabel('Q')
			plt.title('Off-policy Q(.2,.02,%d)'%i)
			plt.show()
		
		fig, ax = plt.subplots()
		ax.matshow(policy, cmap='seismic')
		ax.set_xticks(np.arange(0,np.shape(policy)[1],step=10))
		ax.set_xticklabels(np.round(np.arange(-0.07,0.07,self.res[1]*10),3))
		ax.set_yticks(np.arange(0,np.shape(policy)[0],step=10))
		ax.set_yticklabels(np.round(np.arange(-1.2,0.5,self.res[0]*10),2))
		plt.title('pi*')
		plt.show()

	'''
	TD Policy Iteration
	Flags: on : on vs. off policy learning
	Returns: policy that minimizes Q wrt to controls
	'''
	def _td_policy_iter(self, policy_type=True):
		#on-policy iteration
		if policy_type:
			width = np.shape(self.Q)[0]
			length = np.shape(self.Q)[1]
			Q = np.ones((width,length,3))
			i = 0
			# Q_1 = np.copy(self.Q[24,14,:])
			# Q_2 = np.copy(self.Q[14,12,:])
			# Q_3 = np.copy(self.Q[28,18,:])
			while (np.amax(abs(Q-self.Q)) > self.threshold):
				i+=1
				Q = np.copy(self.Q)
				best_policy = np.argmin(self.Q,axis=2)
				policy = np.zeros((width,length)).astype(int)
				for m in range(width):
					for n in range(length):
						probability = np.ones(3)*(self.epsilon/3)
						probability[best_policy[m,n]] = 1 - self.epsilon + (self.epsilon/3)
						policy[m,n] = np.random.choice([0,1,2],p=probability)
				traj = self.rollout(self.env,policy)
				for j in range(len(traj)-1):
					x = np.int((traj[j][0][0]+1.2)/self.res[0])
					y = np.int((traj[j][0][1]+0.07)/self.res[1])
					u = traj[j][1]
					stage_cost = -traj[j][2]
					x_next = np.int((traj[j+1][0][0]+1.2)/self.res[0])
					y_next = np.int((traj[j+1][0][1]+0.07)/self.res[1])
					u_next = traj[j+1][1]
					self.Q[x,y,u] = self.Q[x,y,u]+self.alpha*(stage_cost+self.gamma*self.Q[x_next,y_next,u_next]-self.Q[x,y,u])
				# Q_1 = np.vstack((Q_1,self.Q[24,14,:]))
				# Q_2 = np.vstack((Q_2,self.Q[14,12,:]))
				# Q_3 = np.vstack((Q_3,self.Q[28,18,:]))
			self.converge = True
			policy = np.argmin(self.Q,axis=2) 
			#self.visualize(Q_1,Q_2,Q_3,policy)
			print('Total number of iterations = ' + str(i))
			return policy

		#off-policy iteration
		else:
			width = np.shape(self.Q)[0]
			length = np.shape(self.Q)[1]
			Q = np.ones((width,length,3))
			i = 0
			# Q_1 = np.copy(self.Q[24,14,:])
			# Q_2 = np.copy(self.Q[14,12,:])
			# Q_3 = np.copy(self.Q[28,18,:])
			while (np.amax(abs(Q-self.Q)) > self.threshold):
				i+=1
				Q = np.copy(self.Q)
				best_policy = np.argmin(self.Q,axis=2)
				policy = np.zeros((width,length)).astype(int)
				for m in range(width):
					for n in range(length):
						probability = np.ones(3)*(self.epsilon/3)
						probability[best_policy[m,n]] = 1 - self.epsilon + (self.epsilon/3)
						policy[m,n] = np.random.choice([0,1,2],p=probability)
				traj = self.rollout(self.env,policy)
				for j in range(len(traj)-1):
					x = np.int((traj[j][0][0]+1.2)/self.res[0])
					y = np.int((traj[j][0][1]+0.07)/self.res[1])
					u = traj[j][1]
					stage_cost = -traj[j][2]
					x_next = np.int((traj[j+1][0][0]+1.2)/self.res[0])
					y_next = np.int((traj[j+1][0][1]+0.07)/self.res[1])
					u_next = traj[j+1][1]
					self.Q[x,y,u] = self.Q[x,y,u]+self.alpha*(stage_cost+self.gamma*np.amin(self.Q[x_next,y_next,:])-self.Q[x,y,u])
				# Q_1 = np.vstack((Q_1,self.Q[24,14,:]))
				# Q_2 = np.vstack((Q_2,self.Q[14,12,:]))
				# Q_3 = np.vstack((Q_3,self.Q[28,18,:]))
			self.converge = True
			policy = np.argmin(self.Q,axis=2) 
			#self.visualize(Q_1,Q_2,Q_3,policy)
			print('Total number of iterations = ' + str(i))
			return policy


	'''
	Sample trajectory based on a policy
	'''
	def rollout(self, env, policy=None, render=False):
		traj = []
		t = 0
		done = False
		c_state = env.reset()
		if policy is None:
			while not done and t < 200:
				action = env.action_space.sample()
				if render:
					env.render()
				n_state, reward, done, _ = env.step(action)
				traj.append((c_state, action, reward))
				c_state = n_state
				t += 1

			env.close()
			return traj

		else:
			Total_dist = 0
			while not done and t < 200:
				x = np.int((c_state[0] + 1.2)/self.res[0])
				y = np.int((c_state[1] + 0.07)/self.res[1])
				action = policy[x,y]
				if render:
					env.render()
				n_state, reward, done, _ = env.step(action)
				traj.append((c_state, action, reward))
				Total_dist+=abs(c_state[0]-n_state[0])
				c_state = n_state
				t += 1

			if (self.converge):
				print('Total horizontal distance traveled = '+str(Total_dist))
			env.close()
			return traj


if __name__ == '__main__':
	env = gym.make('MountainCar-v0')
	res = [0.05, 0.005]		#discretization resolution
	gamma = 0.9 			#discount factor
	alpha = 0.2 			#learning rate
	epsilon = 0.3			#greedy probability
	threshold = 0.0001		#convergence threshold
	policy_type = True		#True -- on-policy		False -- off-policy
	planner = Planner(res,gamma,alpha,epsilon,threshold,env)
	policy = planner(policy_type)
	traj = planner.rollout(env, policy, render=True)
	#print(traj)