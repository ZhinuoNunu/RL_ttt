import numpy as np

# environment definition
class CrossTicTacToe:
    def __init__(self):
        self.board_size = 12
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.valid_regions = self._get_valid_regions()
        self._init_board()
        
    def _get_valid_regions(self):
        regions = [
            (slice(0, 4), slice(4, 8)),   # top
            (slice(8, 12), slice(4, 8)),  # bottom
            (slice(4, 8), slice(0, 4)),   # left
            (slice(4, 8), slice(8, 12)),  # right
            (slice(4, 8), slice(4, 8)),   # center
        ]
        valid = []
        for r_slice, c_slice in regions:
            for r in range(r_slice.start, r_slice.stop):
                for c in range(c_slice.start, c_slice.stop):
                    valid.append((r, c))
        return valid
    
    def _init_board(self):
        self.board.fill(-1)
        for (r, c) in self.valid_regions:
            self.board[r, c] = 0
            
    def get_valid_actions(self):
        return [(r, c) for (r, c) in self.valid_regions if self.board[r, c] == 0]
    
    def is_valid_position(self, pos):
        row, col = pos
        if 0 <= row < 12 and 0 <= col < 12:
            tmp = [row, col]
            if tmp in self.valid_regions:
                return True
        return False
    
    def check_win(self, player):
        # check the 4-in-a-row win condition
        directions_hv = [(0,1), (1,0)]
        directions_diag = [(1,1), (1,-1)]
        for r in range(self.board_size):
            for c in range(self.board_size):
                for dr, dc in directions_hv:
                    if all(0 <= r+i*dr < self.board_size and 
                           0 <= c+i*dc < self.board_size and
                           self.board[r+i*dr][c+i*dc] == player
                           for i in range(4)):
                        return True
                    
        for r in range(self.board_size):
            for c in range(self.board_size):
                for dr, dc in directions_diag:
                    if all(0 <= r+i*dr < self.board_size and 
                           0 <= c+i*dc < self.board_size and
                           self.board[r+i*dr][c+i*dc] == player
                           for i in range(5)):
                        return True
        return False
    
    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, {}
            
        r, c = action
        valid_actions = self.get_valid_actions()
        reward = 0
        
        # action execution logic
        # if np.random.rand() < 0.5:  # 50% probability execute the original action
        if True:
            if (r, c) in valid_actions:
                self.board[r, c] = self.current_player
            else:
                self.done = True
                return self.board.copy(), -1, True, {}
        else:  # random select the adjacent position
            candidates = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        candidates.append((nr, nc))
                        
            if candidates:
                nr, nc = random.choice(candidates)
                if (nr, nc) in valid_actions:
                    self.board[nr, nc] = self.current_player
                else:
                    self.current_player = 3 - self.current_player
                    return self.board.copy(), 0, False, {}
        
        # check the win
        if self.check_win(self.current_player):
            reward = 1 if self.current_player == 1 else -1
            self.done = True
            self.winner = self.current_player
        else:
            reward = -0.01  # step reward
            
        self.current_player = 3 - self.current_player
        return self.board.copy(), reward, self.done, {}
    
    def reset(self):
        self._init_board()
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()