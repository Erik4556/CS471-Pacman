# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        sucessorPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print('successorGameState\n'+str(successorGameState))
        # print('newPos\n' + str(newPos))
        # print('newFood\n' + str(newFood))
        # print('newGhostStates\n' + str(newGhostStates))
        # print('newScaredTimes\n' + str(newScaredTimes))
        # print('action\n'+str(action))
        # print('successorGameState.getScore()\n' + str(successorGameState.getScore()))

        # We need to consider the distance from the agent to the nearest food pellet, and the nearest ghost

        sucessorPos = list(successorGameState.getPacmanPosition())
        dist = -99999999 # Default worst case value that we will change later
        curFoodList = currentGameState.getFood().asList()
        for x in curFoodList: # For each power pellet
            # Catch for being on top of a power pellet, to not consider that pellet as it is about to be removed
            if manhattanDistance(currentGameState.getPacmanPosition(), x) == 0:
                continue
            # Calculate the negative (since we start with a negative value) manhattan distance of the new position to the pellet
            if (-1*manhattanDistance(sucessorPos, x)) > dist: # If that new distance is an improvement from before, keep it
                dist = (-1*manhattanDistance(sucessorPos, x))
        # Catch for pacman heading into a ghost
        for x in newGhostStates: # For each ghost
            # if pacman is about to run into a ghost, RUN.
            # "RUN" is defined as returning the worst possible result, to indicate that option as bad
            if list(x.getPosition()) == sucessorPos:
                return -99999999

        # Finally return the dist if it hasn't already caught on the ghost avoidance loop
        return dist


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actionList = gameState.getLegalActions(0)
        pacmanAgentIndex = 0
        ghostAgentIndices = list(range(1,gameState.getNumAgents())) # List of each agent index for looping
        count = util.Counter()
        agentEnd = gameState.getNumAgents()-1 # Last agent in the list

        def maximizer(curState, agentIndex, depth):

            ghostActions = curState.getLegalActions(agentIndex)
            maxDepth = self.depth  # Quantifying the end of the tree so we know when we reached a leaf node
            weight = -99999999 # Worst case starting value to be changed in the code
            if depth == maxDepth: # If we are at a leaf node
                return self.evaluationFunction(curState) # evaluate the state of this leaf node
            # Otherwise, we progress the tree until the above condition is reached
            if len(ghostActions) != 0:
                for x in ghostActions:
                    if weight >= minimizer(curState.generateSuccessor(agentIndex, x), agentIndex+1, depth):
                        weight = weight
                    else:
                        weight = minimizer(curState.generateSuccessor(agentIndex, x), agentIndex+1, depth)
                return weight
            else:
                # if there are no legal actions left then evaluate at the last known state
                return self.evaluationFunction(curState)

        def minimizer(curState, agentIndex, depth):
            ghostActions = curState.getLegalActions(agentIndex)
            weight = 999999999 # Worst case starting value to be changed in the code
            if len(ghostActions) != 0:
                if agentIndex == agentEnd: # If we've reached the last ghost, we maximise
                    for x in ghostActions: # For each legal action in the current position
                        temp = maximizer(curState.generateSuccessor(agentIndex, x), pacmanAgentIndex, depth+1)
                        if weight < temp:
                            weight = weight
                        else:
                            weight = temp
                else: # Otherwise, we continue to minimize
                    for x in ghostActions: # For each legal action in the current position
                        temp = minimizer(curState.generateSuccessor(agentIndex, x), agentIndex+1, depth)
                        if weight < temp:
                            weight = weight
                        else:
                            weight = temp
                return weight
            else:
                # if there are no legal actions left then evaluate at the last known state
                return self.evaluationFunction(curState)


        # Executing the minimizer for all possible actions
        for x in actionList:
            tempState = gameState.generateSuccessor(pacmanAgentIndex,x)
            count[x] = minimizer(tempState,1,0)
        # print('HELLO THERE')
        # print(count)
        return count.argMax()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # For this problem we will be reusing the majority of our work from question 2, but we will be
        # implementing alpha-beta pruning on top of our existing minimax infrastructure
        actionList = gameState.getLegalActions(0)
        pacmanAgentIndex = 0
        ghostAgentIndices = list(range(1,gameState.getNumAgents())) # List of each agent index for looping
        count = util.Counter()
        agentEnd = gameState.getNumAgents()-1 # Last agent in the list

        def maximizer(curState, agentIndex, alpha, beta, depth):

            ghostActions = curState.getLegalActions(agentIndex)
            maxDepth = self.depth  # Quantifying the end of the tree so we know when we reached a leaf node
            weight = -99999999 # Worst case starting value to be changed in the code
            if depth == maxDepth: # If we are at a leaf node
                return self.evaluationFunction(curState) # evaluate the state of this leaf node
            # Otherwise, we progress the tree until the above condition is reached
            if len(ghostActions) != 0:
                for x in ghostActions:
                    if weight >= minimizer(curState.generateSuccessor(agentIndex, x), agentIndex+1, alpha, beta, depth):
                        weight = weight
                    else:
                        weight = minimizer(curState.generateSuccessor(agentIndex, x), agentIndex+1, alpha, beta, depth)
                    if weight > beta:
                        return weight
                    if alpha < weight:
                        alpha = weight
                return weight
            # if there are no legal actions left then evaluate at the last known state
            # Fall through into this return
            return self.evaluationFunction(curState)

        def minimizer(curState, agentIndex, alpha, beta, depth):
            ghostActions = curState.getLegalActions(agentIndex)
            weight = 999999999 # Worst case starting value to be changed in the code
            if len(ghostActions) != 0:
                if agentIndex == agentEnd: # If we've reached the last ghost, we maximise
                    for x in ghostActions: # For each legal action in the current position
                        temp = maximizer(curState.generateSuccessor(agentIndex, x), pacmanAgentIndex, alpha, beta, depth+1)
                        if weight < temp:
                            weight = weight
                        else:
                            weight = temp
                        if weight < alpha:
                            return weight
                        if beta > weight:
                            beta = weight
                else: # Otherwise, we continue to minimize
                    for x in ghostActions: # For each legal action in the current position
                        temp = minimizer(curState.generateSuccessor(agentIndex, x), agentIndex+1, alpha, beta, depth)
                        if weight < temp:
                            weight = weight
                        else:
                            weight = temp
                        if weight < alpha:
                            return weight
                        if beta > weight:
                            beta = weight
                return weight
            # if there are no legal actions left then evaluate at the last known state
            # Fall through into this return
            return self.evaluationFunction(curState)

        endWeight = -999999999
        alpha = -999999999
        beta = 999999999

        # Executing the minimizer for all possible actions
        for x in actionList:
            tempState = gameState.generateSuccessor(pacmanAgentIndex,x)
            endWeight = minimizer(tempState, 1, alpha, beta, 0,)
            count[x] = endWeight
            if alpha < endWeight:
                alpha = endWeight
        # print('HELLO THERE')
        # print(count)
        return count.argMax()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Again, we use the fundamental foundation built in Q2 for Q4, however here we modify our minimizer function
        # to serve the purpose of finding the expected value
        actionList = gameState.getLegalActions(0)
        pacmanAgentIndex = 0
        ghostAgentIndices = list(range(1,gameState.getNumAgents())) # List of each agent index for looping
        count = util.Counter()
        agentEnd = gameState.getNumAgents()-1 # Last agent in the list
        def maximizer(curState, agentIndex, depth):

            ghostActions = curState.getLegalActions(agentIndex)
            maxDepth = self.depth  # Quantifying the end of the tree so we know when we reached a leaf node
            weight = -99999999  # Worst case starting value to be changed in the code
            if depth == maxDepth:  # If we are at a leaf node
                return self.evaluationFunction(curState)  # evaluate the state of this leaf node
            # Otherwise, we progress the tree until the above condition is reached
            if len(ghostActions) != 0:
                for x in ghostActions:
                    if weight >= minimizer(curState.generateSuccessor(agentIndex, x), agentIndex + 1, depth):
                        weight = weight
                    else:
                        weight = minimizer(curState.generateSuccessor(agentIndex, x), agentIndex + 1, depth)
                return weight
            else:
                # if there are no legal actions left then evaluate at the last known state
                return self.evaluationFunction(curState)

        def minimizer(curState, agentIndex, depth):
            ghostActions = curState.getLegalActions(agentIndex)
            weight = 0 # Starting value of zero to be incremented below
            if len(ghostActions) != 0:
                if agentIndex == agentEnd: # If we've reached the last ghost, we maximise
                    for x in ghostActions: # For each legal action in the current position
                        temp = (float(1.0) / len(ghostActions))*maximizer(curState.generateSuccessor(agentIndex, x), pacmanAgentIndex, depth+1)
                        weight = weight + temp
                else: # Otherwise, we continue to minimize
                    for x in ghostActions: # For each legal action in the current position
                        temp = (float(1.0) / len(ghostActions))*minimizer(curState.generateSuccessor(agentIndex, x), agentIndex+1, depth)
                        weight = weight + temp
                return weight
            else:
                # if there are no legal actions left then evaluate at the last known state
                return self.evaluationFunction(curState)

        # Executing the minimizer for all possible actions
        for x in actionList:
            tempState = gameState.generateSuccessor(pacmanAgentIndex,x)
            count[x] = minimizer(tempState,1,0)
        # print('HELLO THERE')
        # print(count)
        return count.argMax()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: We use a list comprehension to form a list of all manhattan distances from the agent to the food
    # pellet, then take their negatives, and find the highest value, which we add to the current score
    # This allows us to evaluate the score that would come from this given state, and we can simply take the
    # largest value to determine the optimal move
    """
    "*** YOUR CODE HERE ***"

    if currentGameState.getFood().asList() == []: # Null list catch if there is no food on the board
        return currentGameState.getScore()
    else:
        return max([manhattanDistance(currentGameState.getPacmanPosition(),x) * -1
                    for x in currentGameState.getFood().asList()]) + currentGameState.getScore()




# Abbreviation
better = betterEvaluationFunction
