#   2048 Bot Using FFNN
#
#   Description:
#   Two main functionalities:
#   1. Scan a folder containing images of the 2048 game and output the state of the games
#      to a text file called "eval_states.txt" and show the number of correct and attempts.
#   2. Scan a folder containing folders of images and utilize a SGD image classifier to 
#      correctly classify the images.
#
#   Usage:
#   python scan.py -m [scan, train]
#
#   Author
#   Alexandru Micsoniu     
#

# Class Imports
import board_class
# Other Imports
import pytesseract
import cv2
import numpy as np
import os
import sys
import time
import math
import argparse
import random

# Imports specifically used for image classification
import joblib
from skimage.io import imread
from skimage.transform import resize
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
import skimage

import pandas as pd
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import classification_report

from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale

from joblib import dump, load
import PIL.Image

###################################
##### START OF BOARD SCANNING #####
###################################

# Global Variables
TILE_KEYS = {}

# Function to obtain the clean canny edges of an image
def load_images(mode, directory):
    print("[+] Load Images")
    train_images = []
    test_images = []
    scan_images = []
    if mode == "scan":
        for root, dirs, files in os.walk(directory):
            for file in files:
                scan_images.append(os.path.join(root, file))
        scan_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        #print("Scan images:", scan_images)
        return scan_images
    else:
        scan_images.append(directory)
        return scan_images   

# Function to find the average rgb values of images
def find_average_rgb(image, img_keys):
    #print("[+] Find Average RGB")
    # If no image is passed, assume this is first run and populate TITLE_KEYS dictionary
    if image is None:
        # Look inside the keys folder for png files, find the average rgb values of each image, 
        # save them to a dictionary using the file name as the key, and the average rgb value as the value.
        for root, dirs, files in os.walk(img_keys):
            for file in files:
                img = cv2.imread(os.path.join(root, file))
                avg_rgb = np.average(img, axis=0)
                avg_rgb = np.average(avg_rgb, axis=0)
                avg_rgb = np.average(avg_rgb, axis=0)
                TILE_KEYS[file] = avg_rgb
    else:
        #print("Image:", image)
        image = cv2.imread(image)
        # If an image is passed, find the average rgb values of that image
        avg_val = np.average(image, axis=0)
        avg_val = np.average(avg_val, axis=0)
        avg_val = np.average(avg_val, axis=0)
        # Go through the dictionary and subtract the average rgb values from the image, and find the closest average rgb value to 0.
        closest_key = ""
        closest_value = 0
        closest_value_list = []
        for key in TILE_KEYS.keys():
            value = 0
            value = avg_val - TILE_KEYS[key]
            closest_value_list.append(value)
        closest_value = min(closest_value_list, key=abs)
        index = closest_value_list.index(closest_value)
        closest_key = list(TILE_KEYS.keys())[index]
        #print("Closest key:", closest_key)
        #print("Closest value:", closest_value)
        return closest_key.split(".")[0]

# Function to determine the state of the game from a game board image
def image_to_state(images, img_keys):
    print("[+] 2048 Board Image to Board State")
    game_board_list = []
    for image in images:
        image = cv2.imread(image)
        cv2.imshow('image', image)
        print("[+] Please press the Enter key to continue through images")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        # use clahe to improve contrast 
        clahe = cv2.createCLAHE(clipLimit = 2) # clip-limit is the contrast limit for localized changes in contrast.
        contrast = clahe.apply(h)
        # merge the channels back together
        limg = cv2.merge((contrast,s,v))
        # use canny
        #canny = cv2.Canny(limg, 20, 110) # Original
        canny = cv2.Canny(limg, 20, 110, apertureSize = 5, L2gradient = True)
        cv2.imshow('canny', canny)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Add a gradient to the canny image to fill in the gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        #kernel = np.ones((3,3),np.uint8)
        gradient = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow('gradient', gradient)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Find the contours from the gradient image
        #cnts = cv2.findContours(gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Original
        cnts = cv2.findContours(gradient , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1] # Original
        
        min_area = 10000 # Original
        max_area = 40000 # Original
        image_number = 0
        file_name_list = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area and area < max_area:
                x,y,w,h = cv2.boundingRect(c)
                ROI = image[y:y+h, x:x+w]
                cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
                file_name = 'ROI_{}.png'.format(image_number)
                file_name_list.append(file_name)
                cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                image_number += 1
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # convert to string, by looking at every other image in file_name_list
        string_list = []
        #for img in range(0, len(file_name_list), 2): # Option for every 2nd image
        for img in range(len(file_name_list)):
            answer = find_average_rgb(file_name_list[img], img_keys)
            string_list.append(answer)
            
        string_list.reverse()
        game_board_list.append(string_list)
    return game_board_list   

# Function to find the log base 2 of values in the game_board_list
def log_base_2(game_board_list):
    print("[+] Log Base 2")
    for list in game_board_list:
        for value in list:
            if int(value) == 0:
                # replace the value with a 0
                list[list.index(value)] = 0
            else:
                # replace the value in the list with the log base 2 of the value. No decimal places
                list[list.index(value)] = int(math.log(int(value), 2))
    return game_board_list

# Function to reformat the game_board_list into a txt file called eval_states.txt, 
# and compare the values to random_states.txt
def reformat_and_compare(game_board_list, compare, eval, states):
    print("[+] Reformat and Compare")
    # if eval exists, delete it
    if os.path.exists(eval):
        os.remove(eval)
    # for each list in the game_board_list, split it into 4 lists of 4 values each.
    eval_game_board_list = [] 
    for list in game_board_list:
        temp_list = []
        for i in range(0, len(list), 4):
            temp_list.append(list[i:i+4])
        eval_game_board_list.append(temp_list)
        # write the temp_list to a txt file
        with open(eval, 'a') as f:
            for sublist in eval_game_board_list:
                f.write("%s " % sublist)
            f.write("\n")
        eval_game_board_list = []
    # compare each row in eval_states.txt to its corresponding row in random_states.txt.
    # If a match is found, increment attempts and correct by 1. No match, increment attempts by 1.
    if compare == "true":
        attempts = 0
        corrects = 0
        line_num = 1
        with open(eval, 'r') as f:
            with open(states, 'r') as g:
                for line1, line2 in zip(f, g):
                    # remove all whitespace from the lines
                    line1 = line1.replace(" ", "")
                    line2 = line2.replace(" ", "")
                    if line1 == line2:
                        #print("Line", line_num, " correct")
                        corrects += 1
                        attempts += 1
                        line_num += 1
                    else:
                        #print("Line", line_num, " incorrect")
                        attempts += 1
                        line_num += 1
        return attempts, corrects

#################################
##### END OF BOARD SCANNING #####
#################################

#########################################
##### START OF IMAGE CLASSIFICATION #####
#########################################
# Main Reference: https://kapernikov.com/tutorial-image-classification-with-scikit-learn/

def resize_all(src, pklname, width = 150, height=None):
    """
    Load images from the source (src), resize them, and write them
    as arrays to a dictionary together with labels and metadata.
    The dictionary is written to a pickle file (pklname).
    """
    print("[+] Resize All")
    height = height if height is not None else width
    data = {}
    data["filename"] = []
    data["data"] = []
    data["label"] = []
    #pklname = f"{pklname}_{width}x{height}px.pkl"
    # read all images in PATH, resize and write to DESTINATION_PATH
    print( "[+] Subdirectories Found:" )
    for subdir in os.listdir(src):
        print(subdir)
        current_path = os.path.join(src, subdir)
        for file in os.listdir(current_path):
            if file[-3:] in {'jpg', 'png'}:
                im = imread(os.path.join(current_path, file))
                im = resize(im, (width, height))
                data['label'].append(subdir)
                data['filename'].append(file)
                data['data'].append(im)
    #joblib.dump(data, pklname)
    return data

def split_data(data):
    """
    Split the data into training and test sets.
    """
    print("[+] Split Data")
    X = np.array(data['data'])
    y = np.array(data['label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
    return X, X_train, X_test, y, y_train, y_test
  
class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
      
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    HOG: Histogram of Oriented Gradients
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])
        
def transform_train_test(X_train, y_train, X_test, y_test):
    print("[+] Transform, Train, & Test")
    # create an instance of each transformer
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(14, 14), 
        cells_per_block=(2,2), 
        orientations=9, 
        block_norm='L2-Hys')
    scalify = StandardScaler()
    
    # call fit_transform on each transform converting X_train step by step
    X_train_gray = grayify.fit_transform(X_train)
    X_train_hog = hogify.fit_transform(X_train_gray)
    X_train_prepared = scalify.fit_transform(X_train_hog)
    
    print("[+] Train SGD Classifier")
    """
    SGD = Stochastic Gradient Descent
    """
    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    sgd_clf.fit(X_train_prepared, y_train)
    
    print ("[+] Testing SGD Classifier")
    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify.transform(X_test_hog)
    
    print ("[+] Results")
    y_pred = sgd_clf.predict(X_test_prepared)
    #print(np.array(y_pred == y_test)[:25])
    #print('')
    print('[+] Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))
    print('')
    dump(sgd_clf, 'sgd_clf.joblib')
    dump(scalify, 'scalify.joblib')
    return y_pred

def classification_report_sgd(y_test, y_pred):
    print ("[+] SGD Classification Report")
    # Classification report for Labels
    print(classification_report(y_test, y_pred))
    
def example_hog():
    img = imread('example_2048.jpg', as_gray=True)
    
    # scale down the image to one third
    img = rescale(img, 1/3, mode='reflect')
    # calculate the hog and return a visual representation.
    img_hog, hog_img = hog(
        img, pixels_per_cell=(14,14), 
        cells_per_block=(2, 2), 
        orientations=9, 
        visualize=True, 
        block_norm='L2-Hys')
    
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(8,6)
    # remove ticks and their labels
    [a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False) 
        for a in ax]
    
    img_hog = np.expand_dims(img_hog, axis=0)
    hog_img = np.expand_dims(hog_img, axis=0)
    
    ax[0].imshow(img.T, cmap='gray')
    ax[0].set_title('example_2048 (grayscale)')
    ax[1].imshow(hog_img.T, cmap='gray')
    ax[1].set_title('example_2048 (hog)')
    plt.show()

def single_classify(sgd_joblib, scalify_joblib, directory, width = 150, height=None):
    clf = load(sgd_joblib)
    scalify = load(scalify_joblib)
    height = height if height is not None else width
    
    img = PIL.Image.open(directory)
    img = img.convert('RGB')
    img.save(directory)
    
    data = {}
    data["data"] = []
    #data["label"] = []
    im = imread(directory)
    im = resize(im, (width, height))
    #data['label'].append(directory)
    data['data'].append(im)
    
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(14, 14), 
        cells_per_block=(2,2), 
        orientations=9, 
        block_norm='L2-Hys')

    img_gray = grayify.transform(np.array(data['data']))
    img_hog = hogify.transform(img_gray)
    img_prepared = scalify.transform(img_hog)
    y_pred = clf.predict(img_prepared)
    return y_pred
    
#######################################
##### END OF IMAGE CLASSIFICATION #####
#######################################
# Links to where the image data was gathered:
# Go boards:   http://tomasm.cz/imago
# Chess boards:https://osf.io/xf3ka/
    
def main():
    print ("[+] Running scan.py")
    eval = "eval_states.txt"
    states = "random_states.txt"
    sgd_joblib = "sgd_clf.joblib"
    scalify_joblib = "scalify.joblib"
    
    # Set Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", dest = "mode", help = "Mode ( scan, train, classify )")
    parser.add_argument("-d", "--directory", dest = "directory", help = "Image Directory")
    parser.add_argument("-c", "--compare", dest = "compare", help = "Compare Mode (true, false)")
    parser.add_argument("-k", "--keys", dest = "keys", help = "Directory of image keys")
    parser.add_argument("-s", "--states", dest = "states", help = "Text file of states")
    arguments = parser.parse_args()
    # Scan Example:       python scan.py -m scan -d photos/2048_boards/ -c false -k keys/
    # Full Scan Example:  python scan.py -m scan -d photos/2048_boards/ -c false -k keys/ -s random_states.txt
    # Train Example:      python scan.py -m train -d photos/ -c false -k keys/
    # Classify Example:   python scan.py -m classify -d photos/2048_boards/0.png -c false -k keys/
    
    # Argument Validation
    if arguments.mode is None:
        print("[-] Error: Mode not found ( -m [scan, train, classify] )")
        return
    if arguments.directory is None or not os.path.isdir(arguments.directory) and not os.path.exists(arguments.directory):
        print("[-] Error: Image directory not found")
        return
    if arguments.compare is None or arguments.compare not in ['true', 'false']:
        print("[-] Error: Compare mode not found ( -c [true, false] )")
        return
    if arguments.keys is None or not os.path.isdir(arguments.keys):
        print("[-] Error: Image key directory not found")
        return
    if arguments.states is None or not os.path.isfile(arguments.states):
        print("[-] Warning: States text file not found, sourcing states from:", states)
    else:
        states = arguments.states    
    
    mode = arguments.mode
    directory = arguments.directory
    compare = arguments.compare
    img_keys = arguments.keys

    if mode == "scan": 
        print("[+] Scan Mode")
    
        scan_images = load_images(mode, directory)
        find_average_rgb(None, img_keys)
        game_board_list = image_to_state(scan_images, img_keys)
        
        #print("Game board list:")
        #for game in game_board_list:
        #    print(game)

        logged_game_board_list = log_base_2(game_board_list)
        #print("Logged game board list:")
        #print(logged_game_board_list)
        if compare == 'true':
            attempts, corrects = reformat_and_compare(game_board_list, compare, eval, states)
            print("[+] Final evaluations:")
            print("[+] Attempts: ", attempts)
            print("[+] Corrects: ", corrects)
            print("[+] Percentage correct: ", 100*corrects/attempts)
        else:
            reformat_and_compare(game_board_list, compare, eval, states)
    
    elif mode == "train":
        print("[+] Train Mode")
        
        #example_hog()
        #sys.exit(0)
        
        data = resize_all(directory, sgd_joblib, width = 150, height = 150)
        
        X, X_train, X_test, y, y_train, y_test = split_data(data)

        y_pred = transform_train_test(X_train, y_train, X_test, y_test)
        
        classification_report_sgd(y_test, y_pred)

    elif mode == "classify":
        print("[+] Classify Mode")
        y_pred = single_classify(sgd_joblib, scalify_joblib, directory, width = 150, height=None)
        print("[+] Predicted label: ", y_pred)
        
        scan_images = load_images(mode, directory)
        find_average_rgb(None, img_keys)
        game_board_list = image_to_state(scan_images, img_keys)
        logged_game_board_list = log_base_2(game_board_list)
        reformat_and_compare(game_board_list, compare, eval, states)

if __name__ == "__main__":
    main() 
