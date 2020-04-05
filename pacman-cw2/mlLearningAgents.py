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
        # keep track of scores while training
        self.won = []
    
    # Accessor functions for the variable episodesSoFar
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
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        # if Directions.STOP in legal:
        #     legal.remove(Directions.STOP)
        # print "Legal moves: ", legal
        # print "Pacman position: ", state.getPacmanPosition()
        # print "Ghost positions:" , state.getGhostPositions()
        # print "Food locations: "
        # print state.getFood()
        # print "Score: ", state.getScore()

        # Update Q-table
        self.updateQTable(state, legal)

        # Act randomly with probability epsilon
        randNum = random.uniform(0, 1)
        if randNum > self.epsilon:
            pick = self.getActionFromQTable(state, legal)
        else:
            # print "Action picked randomly"
            pick = random.choice(legal)

        # Update self.prevState, self.prevAction, self.prevScore
        # for use in next time step
        self.prevState = state
        self.prevAction = pick
        self.prevScore = state.getScore()

        # We have to return an action
        return pick
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        # print "A game just ended!"

        # Update Q-Table
        self.updateQTable(state, [])

        # Resetting for next episode
        self.prevState = None
        self.prevAction = None
        self.prevScore = None
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()

        if self.getEpisodesSoFar() < self.getNumTraining():
            if state.isWin():
                self.won.append(True)
            else:
                self.won.append(False)

            printAfterEps = 100
            if (self.getEpisodesSoFar() + 1) % printAfterEps == 0:
                print "Episodes ", (self.getEpisodesSoFar() + 1) - printAfterEps, " to ", self.getEpisodesSoFar() + 1, \
                    ". Won: ", sum(self.won)
                self.won = []

            # if self.getEpisodesSoFar() > 1990:
            #     print(self.getEpisodesSoFar())

        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)

    # Getter funcion for Q-value. Get's Q(state, action) from Q-table.
    def getQValue(self, state, action):
        # Extract relevant state from state object.
        relState = self.getRelState(state)

        if (relState, action) not in self.qTable:
            self.qTable[(relState, action)] = 0

        return self.qTable[(relState, action)]

    # Get action a such that Q(state, a) is maximum.
    def getActionFromQTable(self, state, legal):
        # print "getActionFromQTable called"
        # qVals = [self.getQValue(relState, action) for action in legal]
        qVals = []
        for action in legal:
            qVals.append(self.getQValue(state, action))
        return legal[argmax(qVals)]

    # Function to update Q-Table using Bellman Equation
    def updateQTable(self, state, legal):
        # print "updateQTable called"

        # Don't update Q-table on first time step of episode
        if not (self.prevState and self.prevAction):
            # print "Not updating q-table because first time step"
            return

        if not legal:
            # If no legal moves, we are stuck or are in a terminal state, so future rewards = 0.
            # print "No legal moves"
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
        learnedValue = reward + gamma * maxQVal
        newQVal = self.getQValue(self.prevState, self.prevAction) * (1 - alpha) + alpha * learnedValue

        # Update q(prevState, prevAction)
        self.qTable[(self.getRelState(self.prevState), self.prevAction)] = newQVal

    # We consider two states to be identical if the positions
    # of Pacman, ghost(s) and food are the same. So we extract
    # state with only this relevant information (relState)
    # from state object.
    # Assume that position of the food is the same in training
    # examples as in test examples.
    def getRelState(self, state):
        # Cast state.getGhostPositions() to tuple since lists are unhashable
        return state.getPacmanPosition(), tuple(state.getGhostPositions())


# Helper function. Returns index of max value.
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]
