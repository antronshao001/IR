#!/bin/bash
# Put your command below to execute your program.
# Replace "./my-program" with the command that can execute your program.
# Remember to preserve " $@" at the end, which will be the program options we give you.
./docLenExtraction.py $@
./feedback.py $@
./invFileHendle.py $@
#./queryHandle.py $@  # MAP evaluation for parameters tuning
./queryOut.py $@