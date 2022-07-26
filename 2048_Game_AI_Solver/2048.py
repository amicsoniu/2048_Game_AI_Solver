#   2048 Bot Using FFNN
#
#   Description:
#   The bot will using image recognition to identify a 2048 game state. It will
#   then use a Feed-Forward Neural Network to predict the next best move from
#   the game state provided, then completing the game for the user to see the
#   outcome.
#
#   Usage:
#   python 2048.py
#
#   Author  Date        Description
#   GSM     MAR-17      Create new file for FFNN
#   GSM     MAR-22      FFNN integration
#   GSM     MAR-24      Merging and moving bug fixes
#   GSM     MAR-29      Baseline tests
#   GSM     APR-05      Experiments
#   GSM     APR-07      More experiments
#   GSM     APR-08      Full run integration
#

#   Class Imports
import board_class          as BoardClass
import neural_network_class as NeuralNetworkClass

#   Basic Imports
import argparse
import numpy
import pandas
import random
import time

#   Constants
max_moves      = 2000
directions     = ["up", "down", "left", "right"]
possible_tiles = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

#   Globals

#   Convert Tile Data From ID to Value
def ConvertTile ( line ):
    #   Variables
    row     = []
    state   = []
    data    = ""

    #   Get ID Data
    for character in line:
        if character.isnumeric ( ) or character == ",":
            data = data + character

    values = data.split ( "," )
    if len ( values ) != 16:
        print( "[-] Error: No State Loaded" )
        exit()
    
    #   Convert ID Data to Array
    for value in values:
        row.append ( int ( value ) )
        if len ( row ) != 0 and len ( row ) % 4 == 0:
            state.append ( row )
            row = []

    #   Convert ID to Value
    for row in range ( 4 ):
        for column in range ( 4 ):
            if state[row][column] != 0:
                state[row][column] = 2 ** int ( state[row][column] )

    return state

#   Get Best Move From Parameter and Weights
def GetBestMove ( board, network ):
    #   Initialize Board State
    network.board = board
    move          = ""
    state         = [0]
    parameter     = [0,0,0,0]

    #   Step Through Simulation
    scores, states, possible, parameters, valid = network.Step ( )

    if valid == True:
        for move in range ( len ( states ) ):
            if possible[move] == False:
                scores[move] = 0

        #   Normalize Scores
        scores = network.NormalizeScores ( scores )

        best       = numpy.max ( scores )
        best       = scores.index ( best )
        state      = states[best]
        move       = "up" if best == 0 else ( "down" if best == 1 else ( "left" if best == 2 else "right" ) )
        parameter  = parameters[best]

        state_data         = pandas.DataFrame ( state )
        state_data.columns = ["", "", "", ""]
        state_data.index   = ["", "", "", ""]

        #   Print Best Possible Move
        #print ( "Direction:", move, scores )

        #   Spawn Next Move
        board.grid = state
        board.Spawn ( type )

    return move, state, scores, parameter, valid

#   Shuffle Board Tiles
def Shuffle ( board ):
    random.shuffle ( board.grid )

#   Fill Board With Random Powers of 2
def Chaotic ( board ):
    for row in range ( 4 ):
        for column in range ( 4 ):
            if board.grid[row][column] == 0:
                tile = random.choice ( possible_tiles )
                board.grid[row][column] = tile

#   Remove Tile On Black Hole Void Space
def BlackHole ( board, row, column ):
    board.grid[row][column] = 0

#    Train Neural Network
def Train ( dataset, type, network, start ):
    print ( "[+] Train Mode" )
    #   Variables
    states       = []
    boards       = []
    counter      = 0
    move_counter = 0
    win          = False

    #   Read Data File
    infile = open ( dataset, "r" )

    #   Sanitize Data
    for line in infile:
        states.append ( ConvertTile ( line ) )

    infile.close ( )

    #   Initialize Board
    for state in states:
        boards.append ( BoardClass.Board ( state ) )

    print ( "[+] Training", len ( boards ), "states" )

    #   Display Board
    for board in boards:
        counter = counter + 1
        move_counter = 0
        win = False
        #   Baseline Test
        if type == "baseline":
            valid = True
            spawn = True
            while numpy.max ( board.grid ) != 2048 and valid == True and spawn == True:
                move_counter = move_counter + 1
                #   Step Through Simulation
                move  = random.choice ( directions )
                valid = board.Move ( move )
                spawn = board.Spawn ( type )
            if numpy.max ( board.grid ) != 2048:
                network.failures = network.failures + 1
            network.runs = network.runs + 1
        #   Normal Training
        else:
            #   Initial Board State
            data         = pandas.DataFrame ( board.grid )
            data.columns = ["", "", "", ""]
            data.index   = ["", "", "", ""]

            #   Get Best Move and State
            network.runs = network.runs + 1
            first_move, state, first_scores, first_parameters, valid = GetBestMove ( board, network )

            #   Perform Experiment
            if type == "shuffle":
                Shuffle ( board )
            elif type == "chaotic":
                Chaotic ( board )
            elif type == "blackhole":
                blackhole_row    = random.randint ( 0, 3 )
                blackhole_column = random.randint ( 0, 3 )

            #   Check Terminal State (2048)
            while numpy.max ( state ) != 2048 and valid == True and move_counter < max_moves:
                move_counter = move_counter + 1
                #   Get Best Move and State
                move, state, scores, parameters, valid = GetBestMove ( board, network )

                #   Perform Experiment
                if type == "shuffle":
                    Shuffle ( board )
                elif type == "chaotic":
                    Chaotic ( board )
                elif type == "blackhole":
                    BlackHole ( board, blackhole_row, blackhole_column )

            network.Adjust ( first_parameters, valid )
            network.NormalizeWeights ( )

        if numpy.max ( state ) == 2048:
            win = True
        print ( "\t[+] Board Complete:", counter, win, move_counter, ( time.time ( ) - start ) )

        #   Handle Memory Leakage
        del board

#   Main
def Main ( ):
    #   Variables
    network = NeuralNetworkClass.NeuralNetwork ( )
    states  = []

    print ( "[+] 2048 AI Bot" )

    #   Set Arguments
    parser = argparse.ArgumentParser ( )
    parser.add_argument ( "-d", "--dataset", dest = "dataset", help = "Dataset of Board States" )
    parser.add_argument ( "-m", "--mode",    dest = "mode",    help = "Mode ( train, scan )" )
    parser.add_argument ( "-t", "--type",    dest = "type",    help = "Type of Test ( baseline, normal, singlespawn, fullspawn, blackhole, chaotic, shuffle )" )
    arguments = parser.parse_args ( )

    #   Argument Validation
    if arguments.dataset is None:
        print ( "[-] Error: Dataset not found ( -d [statefile] )" )
        return

    if arguments.mode is None:
        print ( "[-] Error: Mode not found ( -m [train, scan] )" )
        return

    if arguments.type is None and arguments.mode == "train":
        print ( "[-] Error: Type not found ( -m [baseline, normal, singlespawn, blackhole, chaotic, shuffle] )" )
        return

    #   Start Timer
    start = time.time ( )

    #   Train Dataset
    if arguments.mode == "train":
        print ( "[+] Type:", arguments.type )
        Train ( arguments.dataset, arguments.type, network, start )
    else:
        print ( "[+] Scan Mode" )

        #   Read Data File
        infile = open ( arguments.dataset, "r" )

        #   Sanitize Data
        for line in infile:
            states.append ( ConvertTile ( line ) )

        #   Pick Single Instance
        state = states[0]

        infile.close ( )
        
        #   Initialize Board
        board = BoardClass.Board ( state )

        #   Initial Visual Board State
        data         = pandas.DataFrame ( board.grid )
        data.columns = ["", "", "", ""]
        data.index   = ["", "", "", ""]
        print ( "[+] Loaded State" )
        print ( data, "\n" )

        first_move, state, first_scores, first_parameters, valid = GetBestMove ( board, network )

    if arguments.mode == "train":
        accuracy = ( network.runs - network.failures ) / network.runs * 100
        print ( "[+] Record: " + str ( accuracy ) + "%" )
        print ( "[+] Final Weights:", network.highest_value, network.decending_home_row, network.decending_home_column, network.decending_home_diagonal )

        #   Print Run Time
        total = time.time ( ) - start
        print ( "[+] Run Time: {:.2f} seconds".format ( total ) )
    else:
        print ( "[+] Best Move:", first_move )
        print ( "Score (up, down, left, right):", first_scores )
        print ( "Parameters:", network.highest_value, network.decending_home_row, network.decending_home_column, network.decending_home_diagonal )

    # board.window.mainloop ( )

if __name__ == "__main__":
    Main ( )
