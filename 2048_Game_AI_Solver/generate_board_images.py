#   Generate 2048 Board Images
#
#   Description:
#   Generate visuals for board states
#
#   Usage:
#   python generate_board_images.py -d states
#
#   Author  Date        Description
#   GSM     MAR-31      Creation to train image recognition
#

#   Imports
import board_class as BoardClass
import argparse
import time
from tkinter import *
import capture_class as tkcap
import os
import shutil

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

#   Load Board States
def LoadBoards ( file ):
    #   Variables
    counter = 0
    states  = []
    boards  = []

    #print ( "[+] Load Board States" )

    #   Read Data File
    infile = open ( file, "r" )

    #   Sanitize Data
    for line in infile:
        states.append ( ConvertTile ( line ) )

    infile.close ( )

    #   Initialize Board
    for state in states:
        boards.append ( BoardClass.Board ( state ) )

    return boards

#   Main
def Main ( ):
    # Variables
    counter = 0
    
    # Set Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data",  dest = "infile", help = "Board States")
    parser.add_argument("-f", "--folder",  dest = "folder", help = "Image Folder")
    arguments = parser.parse_args()
    
    # Argument Validation
    if arguments.infile is None:
        print("[-] Error: Board States not found")
        return
    if arguments.folder is None:
        print("[-] Error: Image Folder not found")
        return
    
    boards = LoadBoards(arguments.infile)
    folder = arguments.folder
    
    # Delete all files in DIR
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    print("[+] Generate Board Images")

    print("[+] Generating", len(boards), "images")

    # Generate Board Images
    for board in boards:
        board.UpdateBoard()
        board.window.update()

        # Capture View
        #time.sleep(0.5) # For 1 image at a time
        cap = tkcap.CAP(board.window)
        cap.capture(folder + str(counter) + ".png", overwrite=False, show = False)
        counter = counter + 1
        board.window.destroy()
        #time.sleep(0.5) # For 1 image at a time
        time.sleep(0.2) # For multiple images
        

    #time.sleep(20)
    board.window.mainloop()

if __name__ == "__main__":
    Main()
