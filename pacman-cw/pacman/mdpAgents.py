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

# Unit step for each action. Own vectors (not game.Actions) to keep integer coordinates.
DIRECTION_VEC = {
    Directions.NORTH: (0, 1),
    Directions.SOUTH: (0, -1),
    Directions.EAST: (1, 0),
    Directions.WEST: (-1, 0),
}

# Motion model, taken from api.makeMove: intended move 0.8, each perpendicular slip 0.1.
INTENDED_PROB = api.directionProb
SLIP_PROB = (1 - api.directionProb) / 2

# Rewards R(s). Food dominates the living cost so its pull survives discounting over distance.
R_EMPTY = -0.04
R_FOOD = 10
R_CAPSULE = 10  # Same as food: capsules are not needed to win.
# R_GHOST = -500

# Value iteration parameters.
MAX_ITER = 200   # Safety cap on sweeps per move.
EPSILON = 0.001  # Converged when no utility changes by more than this.
GAMMA = 0.9      # Discount factor.

DEBUG = False  # Print the map at the start of each game.


class MDPAgent(Agent):
    # State = Pacman's cell only. Food (and later ghosts) are folded into R,
    # which is rebuilt every move before re-solving the MDP.

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print("Starting up MDPAgent!")
        name = "Pacman"

    # Runs at the start of every game: build the static model and reset utilities,
    # since the same agent object plays all games of a run.
    def registerInitialState(self, state):
        self.build_map(state)
        self.U = {}

        if DEBUG:
            self.draw_map()

    # States S (open cells) and the transition table P(s'|s,a). The layout never
    # changes during a game, so this is computed once.
    def build_map(self, state):
        self.walls = set(api.walls(state))
        max_x, max_y = api.corners(state)[-1]
        self.width, self.height = max_x + 1, max_y + 1

        self.open_cells = set()
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.walls:
                    self.open_cells.add((x, y))

        self.transition_table = {}
        for cell in self.open_cells:
            for action in DIRECTION_VEC:
                self.transition_table[(cell, action)] = self.get_transition_probs(cell, action)

    # Debug print of the grid, optionally with a value per cell.
    # The top row is printed first, because y=0 is the bottom of the maze.
    def draw_map(self, values=None):
        for y in range(self.height - 1, -1, -1):
            row = ""
            for x in range(self.width):
                if (x, y) in self.walls:
                    row += "     %" if values is not None else "%"
                elif values is not None:
                    row += "%6.1f" % values[(x, y)]
                else:
                    row += " "
            print(row)

    # Deterministic step: the neighbouring cell, or the same cell if a wall blocks it.
    def move(self, cell, direction):
        x, y = cell
        dx, dy = DIRECTION_VEC[direction]
        next_cell = (x + dx, y + dy)

        if next_cell in self.walls:
            return cell
        return next_cell

    # P(s'|s,a) as [(prob, next_cell), ...]. Attempts that land on the same cell
    # (e.g. two slips into walls) have their probabilities summed.
    def get_transition_probs(self, cell, action):
        attempts = [
            (action, INTENDED_PROB),
            (Directions.LEFT[action], SLIP_PROB),
            (Directions.RIGHT[action], SLIP_PROB),
        ]

        outcomes = {}
        for direction, prob in attempts:
            next_cell = self.move(cell, direction)
            outcomes[next_cell] = outcomes.get(next_cell, 0) + prob

        return [(prob, next_cell) for next_cell, prob in outcomes.items()]

    # R(s) for the current move: living cost everywhere, overlaid with food and capsules.
    # Rebuilt every move because food and capsules disappear when eaten.
    def build_rewards(self, state):
        rewards = {}
        food_cells = set(api.food(state))
        capsule_cells = set(api.capsules(state))

        for cell in self.open_cells:
            rewards[cell] = R_EMPTY
            if cell in food_cells:
                rewards[cell] = R_FOOD
            if cell in capsule_cells:
                rewards[cell] = R_CAPSULE

        return rewards

    # Expected utility of an action: sum over s' of P(s'|s,a) * U(s').
    def expected_utility(self, cell, action, U):
        return sum(p * U[next_cell] for p, next_cell in self.transition_table[(cell, action)])

    # Synchronous value iteration (each sweep reads only the previous sweep's U).
    # Warm-started from the last move's utilities, since R changes little between moves.
    def value_iteration(self, rewards):
        U = {s: self.U.get(s, 0) for s in self.open_cells}

        for _ in range(MAX_ITER):
            U_new = {}
            max_change = 0

            for s in self.open_cells:
                # Bellman update: U(s) = R(s) + gamma * max_a sum_s' P(s'|s,a) U(s').
                best = max(self.expected_utility(s, a, U) for a in DIRECTION_VEC)
                U_new[s] = rewards[s] + GAMMA * best
                max_change = max(max_change, abs(U_new[s] - U[s]))

            U = U_new
            if max_change < EPSILON:
                break

        self.U = U
        return U

    # Policy extraction: the legal move with the highest expected utility.
    def best_action(self, cell, U, legal):
        values = []
        for action in legal:
            values.append((self.expected_utility(cell, action, U), action))
        # max compares utilities first; on an exact tie, the action names decide.
        best_value, best = max(values)
        return best

    # This is what gets run in between multiple games
    def final(self, state):
        print("Looks like the game just ended!")

    # Each move: rebuild R, re-solve the MDP, then act on the best expected utility.
    def getAction(self, state):
        # STOP is not in the MDP's action set, so drop it from the choices.
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        rewards = self.build_rewards(state)
        U = self.value_iteration(rewards)
        best = self.best_action(api.whereAmI(state), U, legal)

        return api.makeMove(best, legal)
