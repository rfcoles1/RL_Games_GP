import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.colors as mcolors

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
        :
        #The grace zone defines an area where there can be no walls
        #This makes it easier to ensure the agent/goal is not initialised in a wall
        self.grace_zone = 3
        self.reset()

    def reset(self):
        self.gen_world()
        self.gen_walker()

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

    #returns only the area seen by agent
    def get_local(self):
        x,y = self.walker_pos
        local = np.copy(self.world[x-self.context:x+self.context+1, y-self.context:y+self.context+1])
        local[self.context, self.context] = 2 #agent is defined as 2 in the world
        return local 

    #returns entire world, including the outside walls
    def get_world(self):
        x,y = self.walker_pos
        world = np.copy(self.world)
        world[x, y] = 2 #agent is defined as 2 in the world
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
 
    def is_done(self):
        return np.array_equal(self.walker_pos, self.goal_pos)

    def display(self):
        grid = gs.GridSpec(2,1)
        fig = plt.figure()

        rgbarray = np.vstack([[1,1,1],[0.1,0,0.1],[0.15,0.15,1],[0.75,0.75,0]])
        cmap = mcolors.ListedColormap(rgbarray)
        ax0 = fig.add_subplot(grid[0])
        
        display = np.copy(self.world)%4
        display[self.walker_pos[0], self.walker_pos[1]] = 2  
        display = display[self.context-1:-self.context+1 ,self.context-1:-self.context+1]
        ax0.imshow(display.T,vmin=0, vmax=3, origin = 'lower',cmap = cmap)
        ax0.set_xticks(np.arange(-.5, self.width+1, 1))
        ax0.set_yticks(np.arange(-.5, self.height+1, 1))
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])       
        ax0.grid(alpha=0.5)

        inner = gs.GridSpecFromSubplotSpec(1,2, subplot_spec=grid[1])
        ax1 = fig.add_subplot(inner[0])
        local = self.get_local()%4
        ax1.imshow(local.T,vmin=0, vmax = 3,origin = 'lower', cmap=cmap)
        ax1.set_xticks(np.arange(-.5, self.context*2+1, 1))
        ax1.set_yticks(np.arange(-.5, self.context*2+1, 1))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])       
        ax1.grid(alpha=0.5)

        ax2 = fig.add_subplot(inner[1])
        ax2.axis('off')
        plt.show() 
    

