import numpy as np
from typing import Tuple

class GridWorld:
    def __init__(self, penalty: float = -1.0, start_state: Tuple[int, int] = (0, 0)) -> None:
        """
        Initializes the GridWorld environment.

        Args:
            penalty (float): The penalty for stepping into the negative terminal state.
            start_state (Tuple[int, int]): The starting position of the agent in the grid.
        """
        
        # Define the grid layout: 0: empty cell, 1: positive terminal state, -1: negative terminal state, np.nan: obstacle
        self.grid = np.array([
            [0,  0,      0,       0],        
            [0,  np.nan, 0,  penalty],       
            [0,  0,      0,       1],        
        ], dtype=float)

        # Grid dimensions and initial state
        self._rows, self._cols = self.grid.shape
        self.start_state = start_state
        self.state = self.start_state

        # Set penalty to be used in displaying policy
        self.penalty = penalty
        
        # Direction relations: 0=up/North,1=right/East,2=down/South,3=left/West
        self.actions = [0, 1, 2, 3]
        
        # cost for each step and slip probability (10% chance to slip left, 10% chance to slip right, 80% chance to go intended direction)
        self.step_cost = -0.04
        self.slip_prob = 0.1
        
        # Direction relations: 0=up,1=right,2=down,3=left
        # maps the intended directions to the directions to slip to
        self._left_slip  = {0: 3, 1: 0, 2: 1, 3: 2}
        self._right_slip= {0: 1, 1: 2, 2: 3, 3: 0}

    def reset(self) -> Tuple[int, int]:
        """
        Resets the environment to the starting state.

        Returns:
            state (Tuple[int, int]): The starting position of the agent in the grid.
        """
        self.state = self.start_state
        return self.state

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """
        Checks if the given state is a terminal state.

        Args:
            state (Tuple[int, int]): The state to check.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        val = self.grid[state]
        return not np.isnan(val) and val != 0.0

    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Determines the next state given the current state and action.

        Args:
            state (Tuple[int, int]): current state of the agent
            action (int): action taken by the agent (0=up,1=right,2=down,3=left)

        Raises:
            ValueError: If the action is not valid.
            
        Returns:
            Tuple[int, int]: The next state after taking the action.
        """
        
        # Current position
        r, c = state
        
        if action == 0:      # up
            nr, nc = r + 1, c
        elif action == 1:    # right
            nr, nc = r, c + 1
        elif action == 2:    # down
            nr, nc = r - 1, c
        elif action == 3:    # left
            nr, nc = r, c - 1
        else:
            raise ValueError("action must be 0,1,2,3")

        # Out of bounds or into wall -> stay put
        if not self.in_bounds(nr, nc) or np.isnan(self.grid[nr, nc]):
            return state
        return (nr, nc)

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Takes a step in the environment based on the given action.

        Args:
            action (int): The action taken by the agent (0=up,1=right,2=down,3=left).
            
        Returns:
            Tuple[Tuple[int, int], float, bool]: A tuple containing the next state, reward, and done status.
        """
        
        # If already terminal, stay there
        if self.is_terminal(self.state):
            return self.state, float(self.grid[self.state]), True
        
        # Validate action
        if action not in (0, 1, 2, 3):
            raise ValueError("action must be 0,1,2,3")

        # If already terminal, stay there; no extra reward after entry
        if self.is_terminal(self.state):
            return self.state, 0.0, True

        # Sample slip outcome (80/10/10)
        choices = [self._left_slip[action], action, self._right_slip[action]]
        probs   = [self.slip_prob, 1.0 - 2*self.slip_prob, self.slip_prob]
        actual  = int(np.random.choice(choices, p=probs))

        # Get next state based on actual action taken
        next_state = self.get_next_state(self.state, actual)

        # Determine reward and done status
        if self.is_terminal(next_state):
            reward = float(self.grid[next_state])
            done = True
        else:
            reward = self.step_cost
            done = False

        # Update state
        self.state = next_state
        return next_state, reward, done
    
    def in_bounds(self, r: int, c: int) -> bool:
        """
        Checks if the given row and column indices are within the grid bounds.

        Args:
            r (int): row index
            c (int): column index

        Returns:
            bool: True if within bounds, False otherwise.
        """
        return 0 <= r < self._rows and 0 <= c < self._cols
