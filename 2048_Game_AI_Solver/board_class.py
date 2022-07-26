#   2048 Board
#
#   Description:
#   Based off of Project Gurukul's 2048 implementation
#
#   Usage:
#   import board_class (in python file)
#
#   Author  Date        Description
#   GSM     MAR-17      Creation
#   GSM     MAR-22      Move updates board state
#   GSM     MAR-24      Terminal state integration
#   GSM     APR-07      Single spawn experiment
#

#   Imports
from tkinter import *
from tkinter import messagebox
import random
import numpy
import pandas

#   Constants
background_colours = {
    "2": "#eee4da",
    "4": "#ede0c8",
    "8": "#edc850",
    "16": "#edc53f",
    "32": "#f67c5f",
    "64": "#f65e3b",
    "128": "#edcf72",
    "256": "#edcc61",
    "512": "#f2b179",
    "1024": "#f59563",
    "2048": "#edc22e",
}

colours = {
    "2": "#776e65",
    "4": "#f9f6f2",
    "8": "#f9f6f2",
    "16": "#f9f6f2",
    "32": "#f9f6f2",
    "64": "#f9f6f2",
    "128": "#f9f6f2",
    "256": "#f9f6f2",
    "512": "#776e65",
    "1024": "#f9f6f2",
    "2048": "#f9f6f2",
}

#   Board Class
class Board:
    #   Constructor
    def __init__ ( self, grid ):
        #   Variables
        self.window   = Tk ( )
        self.window.title ( "2048 FFNN Simulation Results" )
        self.game     = Frame ( self.window, bg = "azure3" )
        self.board    = []
        self.grid     = grid or [[0] * 4 for a in range ( 4 )]
        self.compress = False
        self.moved    = False
        self.score    = 0

        #   Create Board
        for row in range ( 4 ):
            rows = []
            for column in range ( 4 ):
                cell = Label ( self.game, text = "", bg = "azure4", font = ( "arial", 22, "bold" ), width = 4, height = 2 )
                cell.grid ( row = row, column = column, padx = 7, pady = 7 )
                rows.append ( cell )
            self.board.append ( rows )

        #   Display Board
        self.game.grid ( )

    #   Reverse Board State
    def Reverse ( self ):
        for cell in range ( 4 ):
            row    = 0
            column = 3

            while row < column:
                self.grid[cell][row], self.grid[cell][column] = self.grid[cell][column], self.grid[cell][row]
                row    = row    + 1
                column = column - 1

    #   Transpose Board State
    def Transpose ( self ):
        self.grid = [list ( a ) for a in zip ( * self.grid )]

    #   Compress Board State
    def Compress ( self ):
        self.compress = False
        temp_cell     = [[0] * 4 for a in range ( 4 )]

        for row in range ( 4 ):
            counter = 0

            for column in range ( 4 ):
                if self.grid[row][column] != 0:
                    temp_cell[row][counter] = self.grid[row][column]

                    if counter != column:
                        self.compress = True

                    counter = counter + 1

        self.grid = temp_cell

    #   Merge Grid
    def Merge ( self ):
        self.merge = False

        for row in range ( 4 ):
            for column in range ( 4 - 1 ):
                if self.grid[row][column] == self.grid[row][column + 1] and self.grid[row][column] != 0:
                    self.grid[row][column] = self.grid[row][column] * 2
                    self.grid[row][column + 1] = 0
                    self.score = self.score + self.grid[row][column]
                    self.merge = True

    #   Spawn Tile TODO: Only Spawns 2
    def Spawn ( self, mode ):
        cells = []

        for row in range ( 4 ):
            for column in range ( 4 ):
                if self.grid[row][column] == 0:
                    cells.append ( ( row, column ) )

        #   No Room to Spawn Tile:
        if len ( cells ) == 0:
            return False

        current_cell = random.choice ( cells )
        row = current_cell[0]
        column = current_cell[1]

        #   Handle Single Spawn Mode
        if mode == "singlespawn":
            self.grid[row][column] = 2
        else:
            #   Spawn New Tile (90% for 2, 10% for 4)
            tile_spawn = random.randint ( 1, 10 )
            if tile_spawn > 9:
                self.grid[row][column] = 4
            else:
                self.grid[row][column] = 2

        return True

    #   Is Board State Locked
    def Mergable ( self ):
        for row in range ( 4 ):
            for column in range ( 3 ):
                if self.grid[row][column] == self.grid[row][column + 1]:
                    return True

        for row in range ( 3 ):
            for column in range ( 4 ):
                if self.grid[row + 1][column] == self.grid[row][column]:
                    return True

        return False

    #   Display Board
    def UpdateBoard ( self ):
        for row in range ( 4 ):
            for column in range ( 4 ):
                if self.grid[row][column] == 0:
                    self.board[row][column].config ( text = "", bg = "azure4" )
                else:
                    self.board[row][column].config ( text = str ( self.grid[row][column] ), bg = background_colours.get ( str ( self.grid[row][column] ) ), fg = colours.get ( str ( self.grid[row][column] ) ) )

        # print ( "Grid: ", self.grid )

    #   Move Board
    def Move ( self, direction ):
        if direction == "up":
            self.Transpose ( )
            self.Compress ( )
            self.Merge ( )
            self.moved = self.compress or self.merge
            self.Compress ( )
            self.Transpose ( )
        elif direction == "down":
            self.Transpose ( )
            self.Reverse ( )
            self.Compress ( )
            self.Merge ( )
            self.moved = self.compress or self.merge
            self.Compress ( )
            self.Reverse ( )
            self.Transpose ( )
        elif direction == "left":
            self.Compress ( )
            self.Merge ( )
            self.moved = self.compress or self.merge
            self.Compress ( )
        elif direction == "right":
            self.Reverse ( )
            self.Compress ( )
            self.Merge ( )
            self.moved = self.compress or self.merge
            self.Compress ( )
            self.Reverse ( )

        #   Check Terminal State (2048)
        if numpy.max ( self.grid ) == 2048:
            return False

        #   Check Terminal State (No Moves Left)
        if not self.Mergable ( ) and numpy.min ( self.grid ) != 0:
            return False

        # self.UpdateBoard ( )

        return True
