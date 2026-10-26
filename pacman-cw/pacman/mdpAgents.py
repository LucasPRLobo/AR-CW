# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util

class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print("Starting up MDPAgent!")
        name = "Pacman"

    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        print("Running registerInitialState for MDPAgent!")
        print("I'm at:")
        print(api.whereAmI(state))

        self.build_map(state)
        self.draw_map()

    def build_map(self, state):
        self.walls = set(api.walls(state))
        self.corners = api.corners(state)
        max_x, max_y = self.corners[-1]
        self.width, self.height = max_x + 1, max_y + 1

        print(f"The map is of size: {self.width}, {self.height}")

        open_cells = set()

        for x in range(self.width):
            for y in range(self.height):                
                coord = (x,y)
                if coord not in self.walls:
                    open_cells.add(coord)

        print(f"Open cells are: \n{open_cells}")

    def draw_map(self, values=None):
      # Print the grid with the top row first, because y=0 is the bottom of the maze.
      for y in range(self.height - 1, -1, -1):
          row = ""
          for x in range(self.width):
              if (x, y) in self.walls:
                  row += "%"
              elif values is not None:
                  row += "%6.1f" % values[(x, y)]
              else:
                  row += " "
          print(row)


        
    # This is what gets run in between multiple games
    def final(self, state):
        print("Looks like the game just ended!")

    # For now I just move randomly
    def getAction(self, state):
        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        # Random choice between the legal options.
        return api.makeMove(random.choice(legal), legal)
