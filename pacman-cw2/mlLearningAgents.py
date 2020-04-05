# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Initialize q-table
        self.qTable = dict()
        # Save previous state, previous action and previous score in
        # order to update Q-table
        self.prevState = None
        self.prevAction = None
        self.prevScore = None

    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    
    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):
        # print "state: ", state

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        print "Legal moves: ", legal
        print "Pacman position: ", state.getPacmanPosition()
        print "Ghost positions:" , state.getGhostPositions()
        print "Food locations: "
        print state.getFood()
        print "Score: ", state.getScore()

        # Update Q-table
        # print 'update q-table: ', self.prevState, self.prevAction, self.prevScore
        self.updateQTable(state, legal)

        # Now pick what action to take. For now a random choice among
        # the legal moves
        # pick = random.choice(legal)

        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > self.epsilon:
            pick = self.getActionFromQTable(state, legal)
        else:
            print "Action picked randomly"
            pick = random.choice(legal)

        # Update self.prevState, self.prevAction, self.prevScore
        # for use in next time step
        self.prevState = state
        self.prevAction = pick
        self.prevScore = state.getScore()

        print ""
        # We have to return an action
        return pick
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        print "A game just ended!"

        self.updateQTable(state, [])

        print "q-table: ", self.qTable
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)

    def getQValue(self, state, action):
        # if (state, action) in self.qTable:
        #     # print "ALREADY IN QTABLE"
        # else:
        #     self.qTable[(state, action)] = 0

        if (state, action) not in self.qTable:
            self.qTable[(state, action)] = 0

        return self.qTable[(state, action)]

    def getActionFromQTable(self, state, legal):
        # print "getActionFromQTable called"
        # qVals = [self.getQValue(state, action) for action in legal]
        qVals = []
        for action in legal:
            qVals.append(self.getQValue(state, action))
        return legal[argmax(qVals)]

    def updateQTable(self, state, legal):
        print "updateQTable called"
        # if not self.prevState:
        #     return

        # Don't update Q-table on first time step of episode
        if not (self.prevState and self.prevAction):
            print "Not updating q-table cuz first time step"
            # print "prevState: ", self.prevState
            # print "prevAction: ", self.prevAction
            # print "prevScore: ", self.prevScore
            return

        if not legal:
            # If no legal moves we are stuck or in a terminal state, so no future rewards.
            print "No legal moves"
            maxQVal = 0
        else:
            # Get max Q Value i.e. max(Q(state, a)) for all a in legal
            # qVals = [self.getQValue(state, action) for action in legal]
            qVals = []
            for action in legal:
                qVals.append(self.getQValue(state, action))
            maxQVal = max(qVals)

        alpha, gamma = self.getAlpha(), self.getGamma()
        reward = state.getScore() - self.prevScore
        # print "Reward is: ", reward
        learnedValue = reward + gamma * maxQVal
        newQVal = self.getQValue(self.prevState, self.prevAction) * (1 - alpha) + alpha * learnedValue
        self.qTable[(self.prevState, self.prevAction)] = newQVal
        print "Updated Q-table: ", self.qTable

# Helper function
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]
