from os import listdir
from os.path import isfile, join
import os
import sys
"""
    ------------ Organizing Your Training Images ------------
    Specify Class label in front of sub-directory name.
    Images belonging to same class do not necessarily be in the same directory
    example: 
    TrainingImgDirectory
      -- 1.Cat
            -- images
            --  ...    
      -- 2.Dog
            -- images
            --  ...    
      -- 3.Frog
            -- images
            --  ...    
      -- 2.Dog(more)
            -- images
            --  ...    
    \nAll cat iamges will be labeled as 1, Dog as 2, Frog as 3, and Dog(more) as 2
    \n** DO NOT PUT ANY OTHER DIRECTORIES OTHER THAN ACTUAL TRAINING CLASSES&IMAGES **


    \n\n------------ Passing Arguments ------------
    arg1 : Root Directory that contains all training class directories
         -->  \"The absolute Path of TrainingImgDirectory\" in the example above
    arg2 : Result file name
         --> your file will be saved in Root Directory
    If you just pass \'default\', it will take working directory as root directory,
     and file will be saved as \"TrainingImgList.txt\" 
"""


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def generate(RootDirectory,SaveFileName):
 
    ImgDirectory=get_immediate_subdirectories(RootDirectory)
 
    text_file = open(RootDirectory+"/"+SaveFileName+".txt", "w")


 
    for classNum in range(len(ImgDirectory)):
        mypath = RootDirectory+"/"+ImgDirectory[classNum]
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        for f in onlyfiles:
            text_file.write(mypath+"/"+f+" "+ImgDirectory[classNum].split('.', 1)[0]+"\n")
 
    text_file.close()

 


def help():
    print("------------ Organizing Your Training Images ------------")
    print("Specify Class label in front of sub-directory name.")
    print("Images belonging to same class do not necessarily be in the same directory")
    print("example: ")
    print("TrainingImgDirectory")
    print("  -- 1.Cat")
    print("        -- images")
    print("        --  ...    ")
    print("  -- 2.Dog")
    print("        -- images")
    print("        --  ...    ")
    print("  -- 3.Frog")
    print("        -- images")
    print("        --  ...    ")
    print("  -- 2.Dog(more)")
    print("        -- images")
    print("        --  ...    ")
    print("\nAll cat iamges will be labeled as 1, Dog as 2, Frog as 3, and Dog(more) as 2")
    print("\n** DO NOT PUT ANY OTHER DIRECTORIES OTHER THAN ACTUAL TRAINING CLASSES&IMAGES **")


    print("\n\n------------ Passing Arguments ------------")
    print("arg1 : Root Directory that contains all training class directories")
    print("     -->  \"The absolute Path of TrainingImgDirectory\" in the example above")
    print("arg2 : Result file name")
    print("     --> your file will be saved in Root Directory")
    print("If you just pass \'default\', it will take working directory as root directory,")
    print(" and file will be saved as \"TrainingImgList.txt\"") 


run=False


if len(sys.argv) is 1:
    rootDirectory= os.getcwd()
    saveFileName = "TrainingImgList"
    run=True

elif sys.argv[1] == "help":
    help()
elif len(sys.argv) is 3:
    rootDirectory= str(sys.argv[1])
    saveFileName = str(sys.argv[2])
    run=True
else:
    help()

if run:
    print("Training Image Directory: " + rootDirectory)
    print("Saving File as : "+saveFileName +".txt")
 
    generate(rootDirectory,saveFileName)
