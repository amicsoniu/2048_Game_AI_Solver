#! /bin/bash
#
#   Details     -
#   Script to Train and Run 2048 Neural Network Simulation
#
#   Usage       -
#   ./run.sh
#
#   Author  Date        Description
#   GSM     APR-08      Creation
#

#   Train 2048 Feed Forward Neural Network
TrainNeuralNetwork ( )
{
    read -p "[+] Enter States File: " statesfile
    read -p "[+] Enter Type (baseline, normal, singlespawn, chaotic, shuffle): " type

    python.exe 2048.py -d $statesfile -m train -t $type
}

#   Train 2048 Board Binary Image CLassifier
TrainImageClassifier ( )
{
    python.exe scan.py -m train -c false -k keys/ -d photos
}

#   Predict Best Move (1 Board)
RunSimulation ( )
{
    #echo "[-] Warning: save your image alone in its own folder"
    read -p "[+] Enter Image path: " board
    python.exe scan.py -m classify -c false -k keys/ -d $board

    python.exe 2048.py -d eval_states.txt -m scan -t normal 
}

#   Main
echo "@--------------------------------------------@"
echo "|            2048 Best Move                  |"
echo "|                                            |"
echo "|   Authors: Garik Smith-Manchip             |"
echo "|            Alexandru Micsoniu              |"
echo "@--------------------------------------------@"
echo ""

PS3="[+] Enter Option: "

COLUMNS=12
select mode in "Train Neural Network" "Train Image Classifier" "Run Simulation" "Quit"
do
    case $mode in
        "Train Neural Network")     TrainNeuralNetwork;;
        "Train Image Classifier")   TrainImageClassifier;;
        "Run Simulation")           RunSimulation;;
        "Quit")                     break;;
        *)                          echo "Invalid: $REPLY";;
    esac
done

echo ""
