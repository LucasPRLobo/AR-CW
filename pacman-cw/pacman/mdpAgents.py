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

# The original skeleton was written by Simon Parsons, based on the code in
# pacmanAgents.py.


# MDPAgent for the 6CCSARDM coursework
from pacman import Directions
from game import Agent
import api
import util

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
R_FOOD = 5
R_CAPSULE = 30  # Above food: a capsule makes ghosts harmless for a while.
R_GHOST = -300

# Value iteration parameters.
MAX_ITER = 200   # Safety cap on sweeps per move.
EPSILON = 0.001  # Converged when no utility changes by more than this.
GAMMA = 0.85     # Discount factor.

# Ghost model. Rewards, GAMMA and these values were tuned by random search on
# seeds 100-109 and checked on unseen seeds 1000-1099.
SCARED_MARGIN = 2   # A scared ghost counts as dangerous once its timer falls to this.
DANGER_RADIUS = 2   # Maze steps around a ghost that get a penalty.
DANGER_DECAY = 0.35  # Penalty at distance d is R_GHOST * DANGER_DECAY**d.

DEBUG = False  # Print the map at the start of each game.


class MDPAgent(Agent):
    # State = Pacman's cell only. Food, capsules and ghosts are folded into R,
    # which is rebuilt every move before re-solving the MDP. Dangerous ghost
    # cells are terminal states.

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

        # Open cells one step away from each cell, used in the BFS danger zone
        self.neighbours = {cell: {self.move(cell, d) for d in DIRECTION_VEC} - {cell} for cell in self.open_cells}

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

    # Cells occupied by ghosts that can kill Pacman. A scared ghost counts as
    # dangerous once its timer is within SCARED_MARGIN of running out.
    # Ghosts can sit halfway between cells, so both cells they overlap are marked.
    def find_ghosts(self, state):
        dangerous_cells = set()
        for (x, y), timer in api.ghostStatesWithTimes(state):
            if timer > SCARED_MARGIN:
                continue  # Safely scared: neutral, not in R.
            dangerous_cells.add((int(x), int(y)))
            dangerous_cells.add((int(x + 0.5), int(y + 0.5)))
        return dangerous_cells

    # Breadth-first search over open cells from start, up to radius steps.
    # Returns {cell: maze distance}, so walls are respected (unlike Manhattan distance).
    def bfs_danger_zones(self, start, radius):
        distances = {start: 0}
        queue = util.Queue()
        queue.push(start)

        while not queue.isEmpty():
            cell = queue.pop()
            if distances[cell] == radius:
                continue

            for n in self.neighbours[cell]:
                if n not in distances:
                    distances[n] = distances[cell] + 1
                    queue.push(n)

        return distances

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

    # R(s) for the current move: living cost everywhere, overlaid with food,
    # capsules and dangerous ghosts, which are also returned as terminal states.
    # Cells near a ghost get a penalty that shrinks with maze distance (reward
    # shaping), so Pacman keeps clear of ghosts that are about to move.
    # Rebuilt every move because food is eaten and ghosts move.
    def build_rewards(self, state):
        rewards = {}
        food_cells = set(api.food(state))
        capsule_cells = set(api.capsules(state))
        terminal = self.find_ghosts(state)

        for cell in self.open_cells:
            rewards[cell] = R_EMPTY
            if cell in food_cells:
                rewards[cell] = R_FOOD
            if cell in capsule_cells:
                rewards[cell] = R_CAPSULE
            if cell in terminal:
                rewards[cell] = R_GHOST

        # Danger zone: keep the strongest penalty when zones from several ghosts overlap.
        for ghost_cell in terminal:
            for danger_cell, d in self.bfs_danger_zones(ghost_cell, DANGER_RADIUS).items():
                if d == 0:
                    continue  # The ghost cell itself already has R_GHOST.
                rewards[danger_cell] = min(rewards[danger_cell], R_GHOST * DANGER_DECAY ** d)

        return rewards, terminal

    # Expected utility of an action: sum over s' of P(s'|s,a) * U(s').
    def expected_utility(self, cell, action, U):
        return sum(p * U[next_cell] for p, next_cell in self.transition_table[(cell, action)])

    # Synchronous value iteration (each sweep reads only the previous sweep's U).
    # Warm-started from the last move's utilities, since R changes little between moves.
    def value_iteration(self, rewards, terminal):
        U = {s: self.U.get(s, 0) for s in self.open_cells}

        for _ in range(MAX_ITER):
            U_new = {}
            max_change = 0

            for s in self.open_cells:
                if s in terminal:
                    # Absorbing state (game over): no future, so U(s) = R(s).
                    U_new[s] = rewards[s]
                else:
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

        rewards, terminal = self.build_rewards(state)
        U = self.value_iteration(rewards, terminal)
        best = self.best_action(api.whereAmI(state), U, legal)

        return api.makeMove(best, legal)
