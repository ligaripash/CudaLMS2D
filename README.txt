LMS Regression using Guided Topological sweep in degenerate cases - 
                 Description of the code
---------------------------------------------------------------
Author: Gil Shapira, Open University Israel
Last updated: October 1, 2015

This readme file describes the LMS regression code that is located for download in http://www.eecs.tufts.edu/r/geometry/lms. It is based on teh topological sweep code that is located in  http://www.eecs.tufts.edu/r/geometry/sweep and includes only minor changes. For more details on the structure of teh code see the README file located in teh later directory.
Additional information about the algorithm and its time complexity is located in this site as well.
The code is written in C++ and does not use any geometrical libraries for computations. Geomview is used to output the data in a geometrical way to the screen.

The structure of the README file is as follows:
1. How to run the code - description of input file and output methods.
2. Known problems

1. How to run the code?
-----------------------
1.1 Compile
----------
To compile simply type "make"

1.2 Input
---------
The input file is "dat.txt".
The first line of the file includes a number representing the number of input data points and the dimension of the points (currently only 2 is acceptable)
The rest of the lines contain the data set itself. Each line is a pair a b that represents a line y=ax+b. The numbers are divided by spaces.
Example of a legal file:
"3 2
 1 1
 4 6
 -0.5 0.5"
This file contains 3 data points.
The data points can also be viewed as points (x,y) in the primal, that are transformed into lines in the dual. The sweep in this case is executed on the set of lines in the dual.
There is no restriction on the number or values of the data set. Lines can share the same slope or can even be identical (share both a and b)

1.3 Output
----------
The code has two different output methods:
a) Geometrical represantation of the LMS line (default).
b) Alph-numeric data showing the values
1.3.1 Geometrical output
------------------------
The Geomview library is used for geometrical output. If you do not have this library installed you can only use the second form output (see 1.3.2).
Run geomview from the current directory (by typing 'geomview'). The name "Topological sweep" should appear in the left hand window. Press "LMS" and the right hand window will present the original data set and the LMS regression slab. 
- The blue points are multiple points. This means that more than 1 point is located in the same exact position.
Note: Make sure that the file .geomview (part of the tar.zip download file) is present in the directory and has appropriate permissions!
1.3.2 Alpha-numeric output
-------------------------
Type "lms 6" from the shell window. The parameter 6 instructs the computer to perform A/N output of teh final result only. 
To display the intermediate values computed suring the run of teh algorithm type "lms 0".
1.3.3 No output
---------------
Type "lms 5". This form can be used if the algorithm is incorporated in another program.

2. Known problems
-----------------
The main problem is due to floating point arithmetic and errors that arise because of it:
The tests that check allignemnt of 3 points sometimes return incorrect answer (FALSE instead of equal etc').
If such an error occurs the sweep will not be able to advance all the way to teh right (because of the that causes a change in teh topological structure of teh arrangement). It will be detected by the algorithm in it's last phase, by verifying of the sweep reached the right most cut or not and an error message will be displayed to the user.