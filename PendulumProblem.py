import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class EnvAnimate:

    #Initialize Inverted Pendulum
    def __init__(self,m,L,g,b,k,r,gamma,dt,n1,n2,nu,Vmax,Umax,sigma,w,algorithm,total_time,start_angle,threshold):
        self.a = m*L*g         
        self.b = b           
        self.k = k           
        self.r = r           
        self.gamma = gamma    
        self.dt = dt   
        self.n1 = n1          
        self.n2 = n2         
        self.nu = nu         
        self.Vmax = Vmax        
        self.Umax = Umax        
        self.sigma = sigma
        self.w = w
        self.V = np.zeros((self.n1,self.n2+1))
        self.t = np.arange(0.0, total_time, self.dt)
        self.threshold = threshold

        cov = np.dot(self.sigma,np.transpose(self.sigma))*self.dt
        self.kernel = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                diff = np.array([[2*np.pi/self.n1*(i-1)],[2*self.Vmax/self.n2*(j-1)]])
                self.kernel[i,j] = (1/np.sqrt(np.linalg.det(cov)*((2*np.pi)**2)))*np.exp(-1/2*np.dot(np.dot(np.transpose(diff),np.linalg.inv(cov)),diff)[0,0])
        self.kernel = self.kernel/np.sum(self.kernel)

        #value iteration
        if algorithm==0:
            self.policy = self.value_iteration()
        #policy iteration
        elif algorithm==1:
            self.policy = self.policy_iteration()
        

        #interpolate the policy to continuous time, generate trajectory
        self.u, self.theta = self.trajectory(start_angle)
        self.x1 = np.sin(self.theta)
        self.y1 = np.cos(self.theta)
        

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])

        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.8, '', transform=self.ax.transAxes)

    def trajectory(self,start_angle):
        curr_angle = start_angle%(2*np.pi)
        current_velocity = 0
        current_control = self.interpolate_control(curr_angle,current_velocity)
        theta = [curr_angle]
        u = [current_control]
        for i in range(np.shape(self.t)[0]-1):
            mu = np.array([[curr_angle],[current_velocity]]) + np.array([[current_velocity],[self.a*np.sin(curr_angle)-self.b*current_velocity+current_control]])*self.dt+np.dot(self.sigma,self.w)
            curr_angle = mu[0,0]%(2*np.pi)
            current_velocity = np.clip(mu[1,0],-self.Vmax,self.Vmax)
            current_control = self.interpolate_control(curr_angle,current_velocity)
            theta += [curr_angle]
            u += [current_control]
        return u, theta

    #visualyze the value of selected states over episode
    def visualize(self,Q_1,Q_2,Q_3,policy):

        plt.plot(Q_1)
        plt.xlabel('iteration')
        plt.ylabel('V')
        plt.title('policy iteration V(3.14,1.0)')
        plt.show()
    

        plt.plot(Q_2)
        plt.xlabel('iteration')
        plt.ylabel('V')
        plt.title('policy iteration V(1.58,1.5)')
        plt.show()


        plt.plot(Q_3)
        plt.xlabel('iteration')
        plt.ylabel('V')
        plt.title('policy iteration V(4.72,-1.5)')
        plt.show()

        #plot the optimal value function
        V = np.roll(self.V, np.int(np.shape(self.V)[0]/2), axis=0)
        fig, ax = plt.subplots()
        ax.matshow(V, cmap='seismic')
        ax.set_xticks(np.arange(0,50,step=10))
        ax.set_xticklabels(np.round(np.arange(-self.Vmax,self.Vmax,step=2),2))
        plt.xlabel('velocity(rad/s)')
        ax.set_yticks(np.arange(0,180,step=18))
        ax.set_yticklabels(np.round(np.arange(-3.14,3.14,step=0.628),2))
        plt.ylabel('position(rad)')
        plt.title('V*')
        plt.show()

        #plot the optimal policy
        fig, ax = plt.subplots()
        policy = np.roll(policy, np.int(np.shape(policy)[0]/2), axis=0)
        ax.matshow(policy, cmap='seismic')
        ax.set_xticks(np.arange(0,50,step=10))
        ax.set_xticklabels(np.round(np.arange(-self.Vmax,self.Vmax,step=2),2))
        plt.xlabel('velocity(rad/s)')
        ax.set_yticks(np.arange(0,180,step=18))
        ax.set_yticklabels(np.round(np.arange(-3.14,3.14,step=0.628),2))
        plt.ylabel('position(rad)')
        plt.title('pi*')
        plt.show()


    def interpolate_control(self,x1,x2):
        x = x1 / (2*np.pi/self.n1)
        y = (x2+self.Vmax)/(2*self.Vmax/self.n2)
        x_floor = np.int(np.floor(x))%self.n1
        x_ceil = np.int(np.ceil(x))%self.n1
        y_floor = np.int(np.floor(y))
        y_ceil = np.int(np.ceil(y))
        x_r = x - x_floor
        y_r = y - y_floor
        if (y_ceil>self.n2 or y_floor<0):
            if y_ceil>self.n2:
                y = self.n2
            else:
                y = 0
            return self.policy[x_floor,y]*(1-x_r)+self.policy[x_ceil,y]*x_r
        else:
            a = np.dot(np.dot(np.array([[1-x_r,x_r]]),np.array([[self.policy[x_floor,y_floor],self.policy[x_floor,y_ceil]],[self.policy[x_ceil,y_floor],self.policy[x_ceil,y_ceil]]])),np.array([[1-y_r],[y_r]]))[0,0]
            return a

    #Gauss-Seidel value iteration algorithm
    def value_iteration(self):
        policy = np.zeros((self.n1,self.n2+1))
        q = 0
        Q_1 = [self.V[90,30]]
        Q_2 = [self.V[45,35]]
        Q_3 = [self.V[135,15]]
        while(True):
            q += 1
            V = np.copy(self.V)
            for i in range(self.n1):
                x1 = i*(2*np.pi/self.n1)
                for j in range(self.n2+1):
                    x2 = j*(2*self.Vmax/self.n2)-self.Vmax
                    value_control = np.zeros(self.nu+1)
                    for k in range(self.nu+1):
                        u = k*(2*self.Umax/self.nu)-self.Umax
                        value_control[k] = self.stage_cost(x1,u)+self.gamma*self.motion_model(x1,x2,u)
                    policy[i,j] = np.argmin(value_control)*(2*self.Umax/self.nu)-self.Umax
                    self.V[i,j] = np.amin(value_control)          
            if (np.amax(abs(V-self.V))<self.threshold):
                break
            print('number of iteration = '+str(q))
            Q_1 = Q_1 + [self.V[90,30]]
            Q_2 = Q_2 + [self.V[45,35]]
            Q_3 = Q_3 + [self.V[135,15]]
        self.visualize(Q_1,Q_2,Q_3,policy)
        return policy


    #policy iteration algorithm
    def policy_iteration(self):
        #Initialize value and policy function
        policy = np.zeros((self.n1,self.n2+1))
        q = 0
        Q_1 = [self.V[90,30]]
        Q_2 = [self.V[45,35]]
        Q_3 = [self.V[135,15]]
        while (True):
            q+=1
            Vk_odd = np.copy(self.V)
            #Policy Evaluation
            while(True):
                V = np.copy(self.V)
                for i in range(self.n1):
                    x1 = i*(2*np.pi/self.n1)
                    for j in range(self.n2+1):
                        x2 = j*(2*self.Vmax/self.n2)-self.Vmax
                        self.V[i,j] = self.stage_cost(x1,policy[i,j])+self.gamma*self.motion_model(x1,x2,policy[i,j])
                if (np.amax(abs(V-self.V)) < self.threshold):
                    break
            
            #Policy Improvement
            for i in range(self.n1):
                x1 = i*(2*np.pi/self.n1)
                for j in range(self.n2+1):
                    x2 = j*(2*self.Vmax/self.n2)-self.Vmax
                    value_control = np.zeros(self.nu+1)
                    for k in range(self.nu+1):
                        u = k*(2*self.Umax/self.nu)-self.Umax
                        value_control[k] = self.stage_cost(x1,u)+self.gamma*self.motion_model(x1,x2,u)
                    policy[i,j] = np.argmin(value_control)*(2*self.Umax/self.nu)-self.Umax

            diff = np.amax(abs(Vk_odd-self.V))
            print('current difference = ' + str(diff))
            Q_1 = Q_1 + [self.V[90,30]]
            Q_2 = Q_2 + [self.V[45,35]]
            Q_3 = Q_3 + [self.V[135,15]]
            print('number of iteration = '+str(q))
            if (diff < self.threshold):
                break
        self.visualize(Q_1,Q_2,Q_3,policy)
        return policy

    
    #Return the stage cost
    def stage_cost(self,x1,u):
        cost = (1-np.exp(self.k*np.cos(x1)-self.k)+self.r/2*(u**2))*self.dt
        return cost
    
    #Return the expected future cost according to current state and control input
    #x1/x2 - physical current state
    #u  - control input
    def motion_model(self,x1,x2,u):
        mu = np.array([[x1],[x2]]) + np.array([[x2],[self.a*np.sin(x1)-self.b*x2+u]])*self.dt+np.dot(self.sigma,self.w)
        mu[0,0] = mu[0,0]%(np.pi*2)
        mu[1,0] = np.clip(mu[1,0],-self.Vmax,self.Vmax)
        x1_grid = np.int(mu[0,0]/(2*np.pi/self.n1))
        y1_grid = np.int((mu[1,0]+self.Vmax)/(2*self.Vmax/self.n2))
        if (x1_grid<1 or x1_grid>=self.n1-1 or y1_grid<1 or y1_grid>=self.n2):
            return self.V[x1_grid,y1_grid]
        else:
            return np.sum(self.V[x1_grid-1:x1_grid+2,y1_grid-1:y1_grid+2]*self.kernel)

    
    #Provide new rollout theta values to reanimate
    def new_data(self, theta):
        self.theta = theta
        self.x1 = np.sin(theta)
        self.y1 = np.cos(theta)
        self.u = np.zeros(1)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2,-2, 2])
        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

    def init(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text

    def _update(self, i):
        thisx = [0, self.x1[i]]
        thisy = [0, self.y1[i]]
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (self.t[i], self.theta[i], self.u[i]))
        return self.line, self.time_text

    def start(self):
        print('Starting Animation')
        print()
        # Set up plot to call animate() function periodically
        self.ani = FuncAnimation(self.fig, self._update, frames=range(len(self.x1)), interval=25, blit=True, init_func=self.init, repeat=False)
        plt.show()


if __name__ == '__main__':
    #hyperparameters
    #############################################################################
    m = 0.1                             #mass of pendulum               (kg)
    L = 1                               #length of pendulum             (m)
    g = 9.8                             #gravity:                       (m/s^2)
    b = 0.1                             #damping&friction factor
    k = 3                               #shape of the cost
    r = 0.001                           #scales control cost
    gamma = 0.9                         #discount factor
    dt = 0.05                           #time step                      (s)
    n1 = 180                            #discretization size of angle
    n2 = 50                             #discretization size of angular velocity
    nu = 50                             #discretization size of control input   
    Vmax = 5                            #max angular velocity        (rad/s)
    Umax = 5                            #max control input           (N)
    #Vmax = np.pi*n2/(n1*dt)            #max angular velocity        (rad/s)
    #Umax = Vmax*nu*m/(n2*dt)           #max control input           (N)
    sigma = np.array([[0.1,0],[0,0.4]]) #gaussian covariance
    w = np.array([[0.1],[0.1]])         #gaussian motion noise
    algorithm = 0                       #0--value iteration / 1--policy iteration
    total_time = 40.0                   #total animation time           (s)
    start_angle = np.pi                 #initial angle                  (rad)
    threshold = 0.00001                 #convergence threshold
    #############################################################################
    animation = EnvAnimate(m,L,g,b,k,r,gamma,dt,n1,n2,nu,Vmax,Umax,sigma,w,algorithm,total_time,start_angle,threshold)
    animation.start()