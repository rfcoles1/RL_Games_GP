import numpy as np
import GPy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.colors as mcolors

def func(x):
    return 1
mf = GPy.core.Mapping(2,1)
mf.f = func
mf.update_gradients = lambda a,b: None

#mf = GPy.core.Mapping(2,1)
#mf.f = lambda x:1
#mf.update_gradients = lambda a,b: 0
#mf.gradients_X = lambda a,b: 0

class Engine(object):
    def __init__(self):

        #The context defines how far the agent can see
        self.context = 3
        
        #set the range on how tall the map can be
        self.min_height = 5
        self.max_height = 6
        #set the ramge on how long the map can be
        self.min_width = 25
        self.max_width = 25
        
        #The grace zone defines an area where there can be no walls
        #This makes it easier to ensure the agent/goal is not initialised in a wall
        self.grace_zone = 3
        self.reset()                                                                                  

    def reset(self):
        self.gen_world()
        self.gen_walker()
        
        #memory for the GP
        self.input_memory = np.array([]).reshape(2,-1).T
        self.output_memory = []
        
        #budget deines how many observations can be taken
        self.budget = 10
        #take intialise observation
        self.add_data() 

    def gen_world(self):
	    #dimensions of the map
        self.height = np.random.randint(self.min_height,self.max_height+1)
        self.width = np.random.randint(self.min_width,self.max_width+1)
        
        #Open spaces in the world are represented by zeros, walls as ones
        #We make the walls as thick as the agent can see
        self.world = np.zeros([self.width+2*self.context, self.height+2*self.context])
        self.world[:,:self.context],self.world[:,-self.context:] = 1,1
        self.world[:self.context,:],self.world[-self.context:,:] = 1,1

        self.final_wall = self.width + self.context #distance value to the furthest wall

        counter = 1 #keeps track of how many columns have passed with no wall
        i = self.context+self.grace_zone #no walls in left-most area where the agent starts
        while i < self.final_wall-self.grace_zone:
            if np.random.power(counter) > 0.5: #a wall is more likely to be placed if there has not been one in a while
                counter = 0

                length = np.random.randint(0,4) #length of the wall to be added
                start = np.random.randint(0,5) #grid-cell in which the first wallplacement will be
                self.world[i, np.arange(start, start+length)%self.height + self.context] = 1

                i += 1#ensures there cannot two walls simultaneously which will likely create a block

            i += 1
            counter += 1

        #generate the position of the goal, the goal will be denoted as a negative one
        self.goal_pos = np.array([np.random.randint(self.final_wall - self.grace_zone, self.final_wall), np.random.randint(self.context,self.height+self.context)])
        self.world[self.goal_pos[0], self.goal_pos[1]] = -1

    def gen_walker(self):
        self.walker_pos = np.array([np.random.randint(self.context,self.context+3), np.random.randint(self.context,self.height+self.context)])

	#returns the ground truth are surrounding the agent
    def get_true_local(self):
        x,y = self.walker_pos
        local = np.copy(self.world[x-self.context:x+self.context+1, y-self.context:y+self.context+1])
        local[self.context, self.context] = 2
        return local 

	#returns the ground truth map for the entire world
    def get_true_world(self):
        x,y = self.walker_pos
        world = np.copy(self.world)
        world[x, y] = 2
        return world 

    def gen_model(self):
        kern = GPy.kern.RBF(input_dim = 2, variance = .1, lengthscale = 1.0)
        self.model = GPy.models.GPRegression(np.array(self.input_memory),\
            np.array(self.output_memory).reshape(-1,1), kern, mean_function=mf)

        self.model.Gaussian_noise.variance = 0
        self.model.Gaussian_noise.variance.fix()
        kern.lengthscale.fix()
        self.model.optimize()   

        self.model_world = np.zeros([self.width+2*self.context, self.height+2*self.context])

        #ToDo - optimise this 
        for i in range(self.width+2*self.context):
            for j in range(self.height+2*self.context):
                self.model_world[i,j] = np.round(self.get_pred([i,j])[0].flatten()[0],0)
   
    def add_data(self):
        x,y = self.walker_pos
        inps = np.mgrid[x-self.context:x+self.context+1:1, y-self.context:y+self.context+1:1].reshape(2,-1).T
        outs = self.world[inps[:,0], inps[:,1]]
        self.input_memory = np.concatenate([self.input_memory, inps])
        self.output_memory = np.concatenate([self.output_memory,outs])
        
        self.gen_model()
        self.get_model_world()

    def get_pred(self, inp):
        inp = np.array(inp).reshape(len(inp)//2,2)
        return self.model._raw_predict(inp)

    #calculates the model for the entire world 
    #is this function necessary?
    def get_model(self):
        inps = np.mgrid[0:self.width+2*self.context:0.1, 0:self.height+2*self.context:0.1]
        dims = np.shape(inps)[1:3]
        inps = inps.flatten().reshape(2,-1)
        world = np.zeros(dims)
        for i in range(dims[0]):
            for j in range(dims[1]):
                world[i,j] = self.get_pred([i*0.1,j*0.1])[0].flatten()[0] 
        
        x,y = self.walker_pos
        world[x*10:(x+1)*10,y*10:(y+1)*10] = 2
        return world 

    #returns the model of the world, but only the area the surrounds the agent
    def get_model_local(self):
        x,y = self.walker_pos
        local = np.copy(self.model_world[x-self.context:x+self.context+1, y-self.context:y+self.context+1])
        local[self.context, self.context] = 2
        return local

    #returns the model of the entire world
    def get_model_world(self):
        x,y = self.walker_pos
        world = np.copy(self.model_world)
        world[x,y] = 2
        return world
    
    def move_left(self):
        if self.world[self.walker_pos[0] -1, self.walker_pos[1]] < 1:
            self.walker_pos[0] -= 1

    def move_right(self):
        if self.world[self.walker_pos[0] +1, self.walker_pos[1]] < 1:
            self.walker_pos[0] += 1
    
    def move_down(self):
        if self.world[self.walker_pos[0], self.walker_pos[1] -1] < 1:
            self.walker_pos[1] -= 1
    
    def move_up(self):
        if self.world[self.walker_pos[0], self.walker_pos[1] +1] < 1:
            self.walker_pos[1] += 1
   
    def no_move(self):
        return
    
    def illuminate(self):
        if self.budget > 0:
            self.add_data()
            self.budget -= 1

    def is_done(self):
        return np.array_equal(self.walker_pos, self.goal_pos)

    def display(self, flag1 = False):
        if flag1 == True:
            grid = gs.GridSpec(5,1)
        else:
            grid = gs.GridSpec(3,1)
        fig = plt.figure()

        rgbarray = np.vstack([[1,1,1],[0.2,0,0.2],[0.25,1,1],[0.75,0.75,0]])
        cmap = mcolors.ListedColormap(rgbarray)
        ccmap = mcolors.LinearSegmentedColormap.from_list('map',rgbarray, 100)

        ax0 = fig.add_subplot(grid[0])
        display = self.get_true_world()%4
        display = display[self.context-1:-self.context+1 ,self.context-1:-self.context+1]
        ax0.imshow(display.T,vmin=0, vmax=3, origin = 'lower',cmap = cmap)
        ax0.set_xticks(np.arange(-.5, self.width+1, 1))
        ax0.set_yticks(np.arange(-.5, self.height+1, 1))
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])       
        ax0.grid(alpha=0.5)
        ax0.set_title('The true map')

        inner = gs.GridSpecFromSubplotSpec(1,2, subplot_spec=grid[1])
        ax1 = fig.add_subplot(inner[0])
        local = self.get_true_local()%4
        ax1.imshow(local.T,vmin=0, vmax = 3,origin = 'lower', cmap=cmap)
        ax1.set_xticks(np.arange(-.5, self.context*2+1, 1))
        ax1.set_yticks(np.arange(-.5, self.context*2+1, 1))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])       
        ax1.grid(alpha=0.5)
        ax1.set_title('True local area')

        ax2 = fig.add_subplot(inner[1])
        local = self.get_model_local()%4
        ax2.imshow(local.T,vmin=0, vmax = 3,origin = 'lower', cmap=cmap)
        ax2.set_xticks(np.arange(-.5, self.context*2+1, 1))
        ax2.set_yticks(np.arange(-.5, self.context*2+1, 1))
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])       
        ax2.grid(alpha=0.5)
        ax2.set_title('Local area from model')
        
        ax3 = fig.add_subplot(grid[2])
        display = self.get_model_world()%4
        display = display[self.context-1:-self.context+1 ,self.context-1:-self.context+1]
        ax3.imshow(display.T,vmin=0, vmax=3, origin = 'lower',cmap = cmap)
        ax3.set_xticks(np.arange(-.5, self.width+1, 1))
        ax3.set_yticks(np.arange(-.5, self.height+1, 1))
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])       
        ax3.grid(alpha=0.5)
        ax3.set_title('The Model')

        if flag1 == True:
            ax4 = fig.add_subplot(grid[3])
            x,y = self.walker_pos
            world = np.zeros([self.width+2*self.context, self.height+2*self.context])
            for i,xy in enumerate(self.input_memory): 
                world[int(xy[0]),int(xy[1])] = self.output_memory[i]
            ax4.imshow(world.T, vmin=0, vmax=3, origin= 'lower', cmap =cmap)
            ax4.set_xticks(np.arange(-.5, self.width+2*self.context+1, 1))
            ax4.set_yticks(np.arange(-.5, self.height+2*self.context+1, 1))
            ax4.set_xticklabels([])
            ax4.set_yticklabels([])       
            ax4.grid(alpha=0.5)
            ax4.set_title('History of observed data')

            ax5 = fig.add_subplot(grid[4])
            display = self.get_model() 
            display = display[(self.context-1)*10:(-self.context+1)*10 ,(self.context-1)*10:(-self.context+1)*10]
            ax5.imshow(display.T,vmin=0,vmax=3, origin='lower',cmap = ccmap)
            ax5.set_xticks(np.arange(-.5, (self.width+1)*10, 10))
            ax5.set_yticks(np.arange(-.5, (self.height+1)*10, 10))
            ax5.set_xticklabels([])
            ax5.set_yticklabels([])       
            ax5.grid(alpha=0.5)
            ax5.set_title('The model with finer discrete points')
        plt.show()
