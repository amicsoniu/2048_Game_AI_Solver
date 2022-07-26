#   Generate 2048 Board States
#
#   Description:
#   Generate a random amount of board states for 2048 to use as training data
#
#   Usage:
#   python generate_boards.py -n instances
#
#   Author  Date        Description
#   GSM     MAR-18      Create to train model
#   GSM     MAR-29      Create new board states
#   GSM     MAR-31      Generate mix of random and new boards
#

#   Basic Imports
import argparse
import random
from random import choices

#   Constants

#   Generate Board States
def GenerateNewBoards ( instances ):
    #   Variables
    grid = []
    data = []

    for a in range ( instances ):
        grid = []
        for b in range ( 4 ):
            row = [0, 0, 0, 0]
            grid.append ( row )
        for tiles in range ( 2 ):
            row    = random.randint ( 0, 3 )
            column = random.randint ( 0, 3 )
            grid[row][column] = 1

        data.append ( grid )

    return data

#   Generate Board States
def Generate ( instances ):
    #   Variables
    grid  = []
    row   = []
    data  = []

    for a in range ( instances ):
        grid = []
        for b in range ( 4 ):
            row = choices ( range ( 0, 11 ), k = 4 )
            grid.append ( row )

        data.append ( grid )

    return data

#   Record Random Board States
def Record ( states, path ):
    #   Variables
    outfile = open ( path, "w" )

    for state in states:
        outfile.write ( str ( state ) + "\n" )

    outfile.close ( )

#   Main
def Main ( ):
    #   Variables
    outfile = "random_states.txt"
    data    = []
    states  = []
    counter = 0
    new_counter = 0

    #   Set Arguments
    parser = argparse.ArgumentParser ( )
    parser.add_argument ( "-n", "--number",  dest = "instances", help = "Board State Instances" )
    parser.add_argument ( "-o", "--outfile", dest = "outfile",   help = "Output File of Board States" )
    parser.add_argument ( "-t", "--type",    dest = "type",      help = "Generation Type ( new, random )" )
    arguments = parser.parse_args ( )

    #   Argument Validation
    if arguments.instances is None:
        print ( "[-] Error: Instances not found" )
        return

    if arguments.type is None:
        print ( "[-] Error: Generation Type not found ( -t [new, random, mix] )" )
        return

    if arguments.outfile is None:
        print ( "[-] Warning: Outfile not found, writing to:", outfile )
    else:
        outfile = arguments.outfile


    print ( "[+] Generating", arguments.instances, "board states" )
    if arguments.type == "random":
        states = Generate ( int ( arguments.instances ) )
    elif arguments.type == "new":
        states = GenerateNewBoards ( int ( arguments.instances ) )
    elif arguments.type == "mix":
        while counter < int ( arguments.instances ):
            counter = counter + 1
            chance  = random.randint ( 1, 100 )
            if chance > 95:
                states.append ( GenerateNewBoards ( 1 ) )
                new_counter = new_counter + 1
            else:
                states.append ( Generate ( 1 ) )

    Record ( states, outfile )

    print ( "[+] Generating complete" )

    if arguments.type == "mix":
        new_states    = new_counter / counter * 100
        random_states = ( counter - new_states ) / counter * 100
        print ( "[+] Distribution (" + str ( counter ) + " states): " + str ( new_states ) + "% new\t" + str ( random_states ) + "% random" )

if __name__ == "__main__":
    Main ( )
