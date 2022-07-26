#   Neural Network
#
#   Description:
#   A Feed-Forward Neural Network that adjusts four parameter weights to train
#   existing board state data to achieve a winning state of 2048
#
#   Author  Date        Description
#   GSM     MAR-22      Creation
#   GSM     MAR-24      Weight adjustment
#   GSM     MAR-29      Test weight adjustment strategies
#   GSM     MAR-31      Test state mixtures (new and random)
#   GSM     APR-04      Training day!
#   GSM     APR-05      Experiments
#


#   Imports
import board_class  as BoardClass
import numpy
import copy
import pandas

#   Constants
weight_adjustment = 0.01
parameter_names   = {
    0: "Highest Value",
    1: "Decending Home Row",
    2: "Decending Home Column",
    3: "Decending Home Diagonal",
}

#   Neural Network Class
class NeuralNetwork:
    #   Constructor
    def __init__ ( self ):
        #   Variables
        self.board                   = []
        self.runs                    = 0
        self.failures                = 0
        self.highest_value           = 1.0124653215125838e-06
        self.decending_home_row      = 0.33795110141201107
        self.decending_home_column   = 0.32308050154346607
        self.decending_home_diagonal = 0.33896738457920134

    def CalculateAdjustment ( self, value, increase ):
        if increase == True:
            # return value + ( value / self.runs )
            return value + ( value * weight_adjustment )
        else:
            # return value - ( value / self.runs / 2 )
            return value - ( value * weight_adjustment )

    #   Adjust Parameter Weights For Outcome
    def Adjust ( self, parameters, outcome ):
        for parameter in range ( len ( parameters ) ):
            if parameters[parameter] == outcome:
                if parameter == 0:
                    self.highest_value = self.CalculateAdjustment ( self.highest_value, True )
                elif parameter == 1:
                    self.decending_home_row = self.CalculateAdjustment ( self.decending_home_row, True )
                elif parameter == 2:
                    self.decending_home_column = self.CalculateAdjustment ( self.decending_home_column, True )
                elif parameter == 3:
                    self.decending_home_diagonal = self.CalculateAdjustment ( self.decending_home_diagonal, True )
            else:
                if parameter == 0:
                    self.highest_value = self.CalculateAdjustment ( self.highest_value, False )
                elif parameter == 1:
                    self.decending_home_row = self.CalculateAdjustment ( self.decending_home_row, False )
                elif parameter == 2:
                    self.decending_home_column = self.CalculateAdjustment ( self.decending_home_column, False )
                elif parameter == 3:
                    self.decending_home_diagonal = self.CalculateAdjustment ( self.decending_home_diagonal, False )

    #   Normalize Scores
    def NormalizeScores ( self, scores ):
        total = numpy.sum ( scores )

        for score in range ( len ( scores ) ):
            scores[score] = scores[score] / total

        return scores

    #   Normalize Weights
    def NormalizeWeights ( self ):
        total = self.highest_value + self.decending_home_row + self.decending_home_column + self.decending_home_diagonal

        self.highest_value           = self.highest_value           / total
        self.decending_home_row      = self.decending_home_row      / total
        self.decending_home_column   = self.decending_home_column   / total
        self.decending_home_diagonal = self.decending_home_diagonal / total

    #   Step One Move Into Futute (Up, Down, Left, Right)
    def Step ( self ):
        #   Variables
        clone            = BoardClass.Board ( self.board.grid )
        grid             = []
        corners          = []
        diagonal         = []
        move_values      = []
        parameter_values = []
        highest_value    = []
        highest_values   = []
        home_corner      = 0
        scores           = []
        states           = []
        possible         = []
        terminals        = []
        parameter        = []
        parameters       = []

        for a in range ( 4 ):
            #   Make Move
            if a == 0:
                valid = clone.Move ( "up" )
            elif a == 1:
                valid = clone.Move ( "down" )
            elif a == 2:
                valid = clone.Move ( "left" )
            elif a == 3:
                valid = clone.Move ( "right" )

            terminals.append ( valid )

            #   Append Moved State
            states.append ( clone.grid )
            possible.append ( clone.moved )

            #   Check Highest Value
            highest_value = numpy.max ( clone.grid )
            highest_values.append ( highest_value )

            #   Get Location of Home Corner (What if never in corner? TODO)
            corners.append ( clone.grid[0][0] )
            corners.append ( clone.grid[0][3] )
            corners.append ( clone.grid[3][0] )
            corners.append ( clone.grid[3][3] )
            home_corner = numpy.max ( corners )
            home_corner = corners.index ( home_corner )
            corners     = []

            #   Check Decending Home Row, Column
            grid = pandas.DataFrame ( clone.grid )

            #   Top Left
            if home_corner == 0:
                row    = grid.iloc[0]
                column = grid[0]
                decending_home_row    = 1 if all ( earlier >= later for earlier, later in zip ( row, row[1:] ) )       else 0
                decending_home_column = 1 if all ( earlier >= later for earlier, later in zip ( column, column[1:] ) ) else 0
            #   Top Right
            elif home_corner == 1:
                row    = grid.iloc[0]
                column = grid[3]
                decending_home_row    = 1 if all ( earlier <= later for earlier, later in zip ( row, row[1:] ) )       else 0
                decending_home_column = 1 if all ( earlier >= later for earlier, later in zip ( column, column[1:] ) ) else 0
            #   Bottom Left
            elif home_corner == 2:
                row    = grid.iloc[3]
                column = grid[0]
                decending_home_row    = 1 if all ( earlier <= later for earlier, later in zip ( row, row[1:] ) )       else 0
                decending_home_column = 1 if all ( earlier >= later for earlier, later in zip ( column, column[1:] ) ) else 0
            #   Bottom Right
            elif home_corner == 3:
                row    = grid.iloc[3]
                column = grid[3]
                decending_home_row    = 1 if all ( earlier <= later for earlier, later in zip ( row, row[1:] ) )       else 0
                decending_home_column = 1 if all ( earlier <= later for earlier, later in zip ( column, column[1:] ) ) else 0

            #   Check Decending Home Diagonal
            if home_corner == 0 or home_corner == 3:
                diagonal.append ( grid[0][0] )
                diagonal.append ( grid[1][1] )
                diagonal.append ( grid[2][2] )
                diagonal.append ( grid[3][3] )
                decending_home_diagonal = 1 if all ( earlier >= later for earlier, later in zip ( diagonal, diagonal[1:] ) ) else 0
            elif home_corner == 1 or home_corner == 2:
                diagonal.append ( grid[0][1] )
                diagonal.append ( grid[1][2] )
                diagonal.append ( grid[2][1] )
                diagonal.append ( grid[3][0] )
                decending_home_diagonal = 1 if all ( earlier <= later for earlier, later in zip ( diagonal, diagonal[1:] ) ) else 0

            #   Gather Scores
            scores.append ( ( decending_home_row * self.decending_home_row + 1 ) + ( decending_home_column * self.decending_home_column + 1 ) + ( decending_home_diagonal * self.decending_home_diagonal + 1 ) )

            #   Gather Parameters
            parameter.append ( highest_value )
            parameter.append ( decending_home_row )
            parameter.append ( decending_home_column )
            parameter.append ( decending_home_diagonal )
            parameters.append ( parameter )

            #   Reset Clone For Next Move
            diagonal  = []
            parameter = []

            #   Handle Memory Leakage
            del clone
            clone = BoardClass.Board ( self.board.grid )

        #   Find Highest Value
        highest_value = numpy.max ( highest_values )
        for value in range ( len ( highest_values ) ):
            if highest_values[value] == highest_value:
                parameters[value][0] = 1
                scores[value] = scores[value] + ( 1 * self.highest_value + 1 )
                #   Make 2048 State Best State
                if highest_value == 2048:
                    scores[value] = 100
            else:
                scores[value] = scores[value] + ( 0 * self.highest_value + 1 )
                parameters[value][0] = 0

        #   Remove Row and Column Indexes
        grid.columns = ["", "", "", ""]
        grid.index   = ["", "", "", ""]

        #   Get Valid States
        valid = numpy.unique ( terminals )

        #   Handle Memory Leakage
        del clone

        #   Terminal State Lose
        if len ( valid ) == 1:
            if valid == False:
                # print ( "[+] \t\t\t\t\t\t\t Lose ✗" )
                self.failures = self.failures + 1
                return scores, states, possible, parameters, False

        #   Ternimal State Win
        if numpy.max ( grid.to_numpy ( ) ) == 2048:
            # print ( "[+] \t\t\t\t\t\t\t Win! ☺" )
            return scores, states, possible, parameters, False

        return scores, states, possible, parameters, True
