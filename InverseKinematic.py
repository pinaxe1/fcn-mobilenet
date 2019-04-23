# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 19:58:24 2019
Source: http://appliedgo.net/roboticarm   © 2016-2019 Christoph Berger
@author: Pinaxe
"""


# Only the plain math package is needed for the formulas. 
import math

# The lengths of the two segments of the robot’s arm. Using the same length for both segments allows the robot to reach the (0,0) coordinate. 
len1 = 80.0
len2 = 80.0

# The law of cosines, transfomred so that C is the unknown. The names of the sides and angles correspond to the standard names in mathematical writing. Later, we have to map the sides and angles from our scenario to a, b, c, and C, respectively. 
def lawOfCosines(a, b, c ):
	return math.acos((a*a + b*b - c*c) / (2 * a * b))

# The distance from (0,0) to (x,y). HT to Pythagoras. 
def distance(x, y) :
	return math.sqrt(x*x + y*y)

# Calculating the two joint angles for given x and y. 
def angles(x, y) :
    global len1,len2
    # First, get the length of line dist. 
    dist = distance(x, y)
    # Calculating angle D1 is trivial. Atan2 is a modified arctan() function that returns unambiguous results. 
    D1 = math.atan2(y, x)
    #D2 can be calculated using the law of cosines where a = dist, b = len1, and c = len2. 
    D2 = lawOfCosines(dist, len1, len2)
    # Then A1 is simply the sum of D1 and D2. 
    A1 = D1 + D2
    # A2 can also be calculated with the law of cosine, but this time with a = len1, b = len2, and c = dist. 
    A2 = lawOfCosines(len1, len2, dist)
    return A1, A2

# Convert radians into degrees. 
def deg(rad) :
	return rad * 180 / 3.1415926


def main() :
	print("Lets do some tests. First move to (5,5):")
	x, y = 5.0, 5.0
	a1, a2 = angles(x, y)
	print("x=%5.2f, y=%5.2f: A1=%5.2f (%5.2f°), A2=%5.2f (%5.2f°)" % (x, y, a1, deg(a1), a2, deg(a2)))

	print("If y is 0 and x = Sqrt(10^2 + 10^2), then alpha should become 45 degrees and beta should become 90 degrees.")
	x, y = math.sqrt(200), 0
	a1, a2 = angles(x, y)
	print("x=%5.2f, y=%5.2f: A1=%5.2f (%5.2f°), A2=%5.2f (%5.2f°)"%(x, y, a1, deg(a1), a2, deg(a2)))

	print("Now let's try moving to (1, 19).")
	x, y = 1, 19
	a1, a2 = angles(x, y)
	print("x=%5.2f, y=%5.2f: A1=%5.2f (%5.2f°), A2=%5.2f (%5.2f°)", x, y, a1, deg(a1), a2, deg(a2))

	print("n extreme case: (20,0). The arm needs to stretch along the y axis.")
	x, y = 20, 0
	a1, a2 = angles(x, y)
	print("x=%5.2f, y=%5.2f: A1=%5.2f (%5.2f°), A2=%5.2f (%5.2f°)", x, y, a1, deg(a1), a2, deg(a2))

	print("And (0,20).")
	x, y = 0, 20
	a1, a2 = angles(x, y)
	print("x=%5.2f, y=%5.2f: A1=%5.2f (%5.2f°), A2=%5.2f (%5.2f°)", x, y, a1, deg(a1), a2, deg(a2))

	print("Moving to (0,0) technically works if the arm segments have the same length, and if the arm does not block itself. Still the result looks a bit weird!?")
	x, y = 0, 0
	a1, a2 = angles(x, y)
	print("x=%5.2f, y=%5.2f: A1=%5.2f (%5.2f°), A2=%5.2f (%5.2f°)", x, y, a1, deg(a1), a2, deg(a2))

	print("What happens if the target point is outside the reach? Like (20,20).")
	x, y = 20, 20
	a1, a2 = angles(x, y)
	print("x=%5.2f, y=%5.2f: A1=%5.2f (%5.2f°), A2=%5.2f (%5.2f°)", x, y, a1, deg(a1), a2, deg(a2))
    
if __name__ == "__main__":
   main()    