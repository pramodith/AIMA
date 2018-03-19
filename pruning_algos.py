#!/usr/bin/env python
from isolation import Board, game_as_text
import sys
import time
import random

# This file is your main submission that will be graded against. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.

# Submission Class 1
class OpenMoveEvalFn:
    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board.

        Args
            param1 (Board): The board and game state.
            param2 (bool): True if maximizing player is active.

        Returns:
            float: The current state's score. Number of your agent's moves.

        """
        moves=game.get_legal_moves()
        # TODO: finish this function!
        # raise NotImplementedError
        if len(moves)==0 and maximizing_player_turn:
            return float("-inf")
        elif len(moves)==0 and not maximizing_player_turn:
            return float("inf")
        else:
            return len(moves)


# Submission Class 2
class CustomEvalFn:
    def __init__(self):
        pass

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Custom evaluation function that acts however you think it should. This
        is not required but highly encouraged if you want to build the best
        AI possible.

        Args
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if maximizing player is active.

        Returns:
            float: The current state's score, based on your own heuristic.

        """
        prev=game.__last_queen_move__
        moves=game.get_legal_moves()
        if len(moves)==0 and maximizing_player_turn:
            return float("-inf")
        elif len(moves)==0 and not game.maximizing_player_turn:
            return float("inf")
        else:
            return len(moves)

class CustomPlayerMinimax:

    def __init__(self, search_depth=3, eval_fn=OpenMoveEvalFn()):
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.oppn_move=None
        self.oppn_board=None

    def move(self, game, legal_moves, time_left):
        best_move, utility = self.minimax(game, time_left, depth=self.search_depth)
        return best_move

    def utility(self, game):
        return self.eval_fn.score(game)

    def minimax(self, game, time_left, depth=3, maximizing_player=True):
        #copy the state of the board and keep track of the last move of the opponent
        self.oppn_board=game.copy()
        self.oppn_move=game.__last_queen_move__
        #initialize the best score to -inf.
        best_val = float("-inf")
        best_move = (-1, -1)
        final_best_move=None
        final_best_val=None
        move_count=0
        moves = game.get_legal_moves()

        #iterate through your search space to score each possible move
        for mv in moves:
            if time_left()>50:
                game.__last_queen_move__=mv
                game.__board_state__[mv[0]][mv[1]] = 'Q'
                move_count += 1
                #obtain the score of the move, keep track of the number of moves so that the max depth parameter is satsified.
                res = self.min_player(game,time_left,move_count)
                move_count -= 1
                game.__last_queen_move__ =self.oppn_move
                game.__board_state__[mv[0]][mv[1]] = 0
                #storing the best move to make and it's score
                if res > best_val:
                    best_val = res
                    best_move = mv

            else:
                break

        return best_move, best_val

    def min_player(self, game,time_left,move_count):
        #the goal is to minimize the opponents score so the opponent's best_val is initialized to inf
        best_val = float("inf")
        self.oppn_move=game.__last_queen_move__
        legal_moves=game.get_legal_moves()
        # if no moves are left the opponent has won
        if len(legal_moves)==0:
            return float("inf")

        #if we  exceed the time limit or the max depth has been reached
        if move_count==self.search_depth or time_left()<50:
            return self.utility(game)
        else:
            #iterate through all possible moves
            for mv in legal_moves:
                game.__last_queen_move__=mv
                game.__board_state__[mv[0]][mv[1]] = 'Q'
                move_count += 1
                res = self.max_player(game,time_left,move_count)
                move_count -= 1
                game.__last_queen_move__=self.oppn_move
                game.__board_state__[mv[0]][mv[1]] = 0
                if res < best_val:
                    best_val = res
            return best_val

    def max_player(self, game,time_left,move_count):
        #same logic as above but from the view of p1.
        best_val = float("-inf")
        self.oppn_move=game.__last_queen_move__
        legal_moves=game.get_legal_moves()
        if len(legal_moves)==0:
            return float("-inf")
        if  time_left()<50 or move_count==self.search_depth:
            return self.utility(game)
        else:
            for mv in legal_moves:
                game.__last_queen_move__=mv
                game.__board_state__[mv[0]][mv[1]] = 'Q'
                move_count += 1
                res = self.min_player(game,time_left,move_count)
                move_count -= 1
                game.__last_queen_move__=self.oppn_move
                game.__board_state__[mv[0]][mv[1]] = 0
                if res > best_val:
                    best_val = res
            return best_val


class CustomPlayer:
    """Player that chooses a move using
    your evaluation function and
    a minimax algorithm
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move."""

    def __init__(self, search_depth=3, eval_fn=OpenMoveEvalFn()):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.oppn_move=None
        self.oppn_board=None
        self.level=0
        self.transposition={}

    def move(self, game, legal_moves, time_left):
        """Called to determine one move by your agent

        Args:
            game (Board): The board and game state.
            legal_moves (dict): Dictionary of legal moves and their outcomes
            time_left (function): Used to determine time left before timeout

        Returns:
            (tuple): best_move
        """

        final_best_move=(-1,-1)
        prev_move=game.__last_queen_move__
        for level in range(1,20):
            #print ""
            if time_left()<50:
                return final_best_move
            best_move=(-1,-1)
            best_val=float("-inf")

            for mv in legal_moves:

                if time_left()<50:
                    return final_best_move
                game.__last_queen_move__=mv
                game.__board_state__[mv[0]][mv[1]] = 'Q'
                tmp,res=self.alphabeta(game,time_left,level-1,alpha=best_val,beta=float("inf"),maximizing_player=False)
                game.__last_queen_move__ =prev_move
                game.__board_state__[mv[0]][mv[1]] = 0
                if res > best_val:
                    best_move= mv
                    best_val=res
            if best_val==float("-inf"):
                return final_best_move
            final_best_move=best_move

        return final_best_move

    def utility(self, game):
        """Can be updated if desired"""
        return self.eval_fn.score(game)

    def alphabeta(self, game, time_left, depth=3, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        legal_moves=self.good_moves(game)
        prev_move=game.__last_queen_move__

        num_moves=len(legal_moves)
        #maximizing player indicates whose turn it is
        if maximizing_player:
            if num_moves==0:
                return prev_move,float("-inf")
            if depth==0 or time_left()<50:
                return prev_move,num_moves
            else:
                best_move=(-1,-1)
                for mv in legal_moves:
                    if time_left()<50:
                        return best_move,alpha
                    game.__last_queen_move__=mv
                    game.__board_state__[mv[0]][mv[1]] = 'Q'
                    #depth parameter decreases by 1 each recursive call to keep track of how deep we are in the search space.
                    tmp,res=self.alphabeta(game,time_left,depth-1,alpha,beta,False)
                    game.__last_queen_move__ =prev_move
                    game.__board_state__[mv[0]][mv[1]] = 0
                    #pruning condition
                    if res >= beta:
                        return mv,beta
                    if res > alpha:
                        alpha=res
                        best_move= mv
                return best_move,alpha

        if not maximizing_player:
            if num_moves==0:
                return prev_move,float("inf")
            if depth==0 or time_left()<50:
                return prev_move,num_moves
            else:
                prev_move=game.__last_queen_move__
                best_move=(-1,-1)
                for mv in legal_moves:
                    if time_left()<50:
                        return best_move,beta
                    game.__last_queen_move__=mv
                    game.__board_state__[mv[0]][mv[1]] = 'Q'
                    tmp,res=self.alphabeta(game,time_left,depth-1,alpha,beta,True)
                    game.__last_queen_move__ =prev_move
                    game.__board_state__[mv[0]][mv[1]] = 0
                    #pruning condition
                    if res <= alpha:
                        return mv,alpha
                    if res < beta:
                        beta=res
                        best_move= mv
                return best_move,beta


    def good_moves(self,game):
        prev_move=game.__last_queen_move__
        moves=[]
        if prev_move==(-1,-1):
            return [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)]
        first_diff=game.width-prev_move[0]
        sec_diff=game.height-prev_move[1]

        for i in range(1,min(first_diff,sec_diff)):
            cord=(prev_move[0]+i,prev_move[1]+i)
            if game.__board_state__[cord[0]][cord[1]]==0:
                moves.append(cord)
            else:
                break

        for i in range(1,first_diff):
            cord=(prev_move[0]+i,prev_move[1])
            if game.__board_state__[cord[0]][cord[1]]==0:
                moves.append(cord)
            else:
                break

        for i in range(1,min(game.height-prev_move[0],1+prev_move[1])):
            cord=(prev_move[0]+i,prev_move[1]-i)
            if game.__board_state__[cord[0]][cord[1]]==0:
                moves.append(cord)
            else:
                break

        for i in range(1,sec_diff):
            cord=(prev_move[0],prev_move[1]+i)
            if game.__board_state__[cord[0]][cord[1]]==0:
                moves.append(cord)
            else:
                break

        for i in range(1,prev_move[1]+1):
            cord=(prev_move[0],prev_move[1]-i)
            if game.__board_state__[cord[0]][cord[1]]==0:
                moves.append(cord)
            else:
                break

        for i in range(1,min(prev_move[0]+1,sec_diff)):
            cord=(prev_move[0]-i,prev_move[1]+i)
            if game.__board_state__[cord[0]][cord[1]]==0:
                moves.append(cord)
            else:
                break

        for i in range(1,prev_move[0]+1):
            cord=(prev_move[0]-i,prev_move[1])
            if game.__board_state__[cord[0]][cord[1]]==0:
                moves.append(cord)
            else:
                break


        for i in range(1,min(prev_move[0],prev_move[1])+1):
            cord=(prev_move[0]-i,prev_move[1]-i)
            if game.__board_state__[cord[0]][cord[1]]==0:
                moves.append(cord)
            else:
                break
        return moves

