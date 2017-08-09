#########
## File Created 2017.08.02
## Double click this file to generate 
## TrainingSetList and TestSetList
## it randomly sort images and make lists out of it
#########
import os
from os import listdir
from os.path import isfile, join
import sys
import random


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def generate(RootDirectory=os.getcwd(),TestSetRatio=20):
 
    ImgDirectory=get_immediate_subdirectories(RootDirectory)
  
    # remove folders that are not classes (starting with non-integer name)  
    ImgDirectory=[x for x in ImgDirectory if (x.split('.', 1)[0]).isdigit()]
    

    Training_text_file = open(RootDirectory+"/TrainingImgList.txt", "w")
    Test_text_file = open(RootDirectory+"/TestImgList.txt", "w")

    countTraining=0
    countTest=0
    print("Generating TrainingSetList and TestSetList in :" + RootDirectory)
    print("Training Set : Test Set = " + str(100-TestSetRatio) +" : "+str(TestSetRatio))
    for classNum in range(len(ImgDirectory)):
        print("Processing image files in "+ ImgDirectory[classNum])
        mypath = RootDirectory+"/"+ImgDirectory[classNum]
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        random.shuffle(onlyfiles)

        testList=onlyfiles[0:int(len(onlyfiles)*TestSetRatio/100)]
        trainList=onlyfiles[int(len(onlyfiles)*TestSetRatio/100):]

        for f in testList:
            Test_text_file.write(mypath+"/"+f+" "+ImgDirectory[classNum].split('.', 1)[0]+"\n")
            countTest+=1
        for f in trainList:
            Training_text_file.write(mypath+"/"+f+" "+ImgDirectory[classNum].split('.', 1)[0]+"\n")
            countTraining+=1
    

    print("# of Training Images : " + str(countTraining))
    print("# of   Test   Images : " + str(countTest)) 

    Training_text_file.close()
    Test_text_file.close()




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
    print("arg2 : Test set Ratio")
    print("     --> Set the amount of test set you want to make (pass argument in percentage)")
    print(" and file will be saved as \"TrainingImgList.txt\"") 


run=False


if len(sys.argv) is 1:
    generate()
elif len(sys.argv) is 3:
    rootDirectory= str(sys.argv[1])
    testSetRatio = str(sys.argv[2])
    generate(rootDirectory,saveFileName)
else:
    help()
     
