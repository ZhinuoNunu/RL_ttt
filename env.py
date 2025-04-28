import numpy as np


# 环境定义
class CrossTicTacToe:
    def __init__(self):
        self.board = np.zeros((12, 12), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.total_regions = range(12)
        self.valid_regions_slice = [
            (slice(0, 4), slice(4, 8)),   # Top
            (slice(8, 12), slice(4, 8)),  # Bottom
            (slice(4, 8), slice(0, 4)),   # Left
            (slice(4, 8), slice(8, 12)),  # Right
            (slice(4, 8), slice(4, 8)),   # Center
        ]
        self.valid_regions = self.get_valid_regions(self.valid_regions_slice)
        self._init_board()

    def get_valid_regions(self, valid_regions_slice):
        valid_regions = []
        for r_slice, c_slice in valid_regions_slice:
            for i in self.total_regions[r_slice]:
                for j in self.total_regions[c_slice]:
                    valid_regions.append([i, j])
        return valid_regions
    
    def _init_board(self):
        self.board.fill(-1)
        for region in self.valid_regions:
            rows, cols = region
            self.board[rows, cols] = 0

    def is_valid_position(self, pos):
        row, col = pos
        if 0 <= row < 12 and 0 <= col < 12:
            tmp = [row, col]
            if tmp in self.valid_regions:
                return True
        return False

    def get_valid_actions(self):
        return [(r, c) for r in range(12) for c in range(12) 
                if self.is_valid_position((r, c)) and self.board[r, c] == 0]

    def check_line(self, player, start_r, start_c, dr, dc, length):
        for i in range(length):
            r = start_r + i*dr
            c = start_c + i*dc
            if not self.is_valid_position((r, c)) or self.board[r, c] != player:
                return False
        return True

    def check_win(self, player):
        cum_reward = 0
        # Check rows and columns for 4 in a row
        for r in range(12):
            for c in range(9):
                if self.check_line(player, r, c, 0, 1, 4):
                    return 2, True
                else:
                    if self.check_line(player, r, c, 0, 1, 3):
                        cum_reward += 0.1
                    
                
        for c in range(12):
            for r in range(9):
                if self.check_line(player, r, c, 1, 0, 4):
                    return 2, True
                else:
                    if self.check_line(player, r, c, 1, 0, 3):
                        cum_reward += 0.1

        # Check diagonals for 5 in a line
        for r in range(8):
            for c in range(8):
                if self.check_line(player, r, c, 1, 1, 5):
                    return 2, True
                else:
                    if self.check_line(player, r, c, 1, 1, 4):
                        cum_reward += 0.1
                        
        for r in range(8):
            for c in range(4, 12):
                if self.check_line(player, r, c, 1, -1, 5):
                    return 2, True
                else:
                    if self.check_line(player, r, c, 1, -1, 4):
                        cum_reward += 0.1
                        
        return cum_reward, False

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, {}

        # Action processing
        r, c = action
        if not self.is_valid_position(action) or self.board[r, c] != 0:
            self.done = True
            return self.board.copy(), -1, True, {}

        # Placement attempt
        # if np.random.rand() < 0.5:
        if np.random.rand() < 2:
            final_pos = action
        else:
            directions = [(-1,-1), (-1,0), (-1,1),
                        (0,-1), (0,1),
                        (1,-1), (1,0), (1,1)]
            candidates = [(r+dr, c+dc) for dr, dc in directions]
            rand_idx = np.random.randint(0, 16)
            final_pos = candidates[rand_idx%8] if rand_idx < 8 else None

        reward = 0
        if final_pos and self.is_valid_position(final_pos) and self.board[final_pos] == 0:
            self.board[final_pos] = self.current_player
            tmp_reward, tmp_done = self.check_win(self.current_player)
            reward = tmp_reward if self.current_player == 1 else -1*tmp_reward
            if tmp_done:
                self.done = True
                self.winner = self.current_player
            else:
                self.current_player = 3 - self.current_player
        else:
            self.current_player = 3 - self.current_player

        return self.board.copy(), reward, self.done, {}

    def reset(self):
        self._init_board()
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()