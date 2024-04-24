# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            count_tracker = util.Counter()
            states = self.mdp.getStates()

            for state in states:
                max = float("-inf")

                for move in self.mdp.getPossibleActions(state):
                    qValue = self.computeQValueFromValues(state, move)

                    if qValue > max:
                        max = qValue

                    count_tracker[state] = max

            self.values = count_tracker

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        moves = self.mdp.getTransitionStatesAndProbs(state, action)
        reward_sum = 0

        for following_state, probability in moves:
            reward = self.mdp.getReward(state, action, following_state)
            reward_sum += probability * (reward + self.discount * self.values[following_state])

        return reward_sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        max = float("-inf")

        for move in self.mdp.getPossibleActions(state):
            qValue = self.computeQValueFromValues(state, move)

            if qValue > max:
                max = qValue
                best_action = move

        return best_action


    def getPolicy(self, state):
        "Returns the policy at the state."
        "*** YOUR CODE HERE ***"
        return self.computeActionFromValues(state)


    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        queue = util.PriorityQueue()
        predecessor_set = {}
        
        for state in self.mdp.getStates():
          
          if not self.mdp.isTerminal(state):

            for move in self.mdp.getPossibleActions(state):
              
              for following_state, probability in self.mdp.getTransitionStatesAndProbs(state, move):

                if following_state in predecessor_set:
                  predecessor_set[following_state].add(state)

                else:
                  predecessor_set[following_state] = {state}
        
        for state in self.mdp.getStates():
          
          if not self.mdp.isTerminal(state):

            values = []

            for move in self.mdp.getPossibleActions(state):
              
              qValue = self.computeQValueFromValues(state, move)
              values.append(qValue)

            difference = abs(max(values) - self.values[state])
            queue.update(state, -difference)

        for i in range(self.iterations):
          
          if queue.isEmpty():
            break
          
          temp = queue.pop()

          if not self.mdp.isTerminal(temp):
            values = []

            for move in self.mdp.getPossibleActions(temp):
              
              qValue = self.computeQValueFromValues(temp, move)
              values.append(qValue)
              
            self.values[temp] = max(values)

            for predecessor in predecessor_set[temp]:

                if not self.mdp.isTerminal(predecessor):
                    values = []

                    for move in self.mdp.getPossibleActions(predecessor):

                        q_value = self.computeQValueFromValues(predecessor, move)
                        values.append(q_value)

                    difference = abs(max(values) - self.values[predecessor])
                    if difference > self.theta:
                        queue.update(predecessor, -difference)