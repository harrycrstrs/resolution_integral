### This code defines two functions for calculating resolution integrals from Edinburgh Pipe Phantom data or similar Ultrasound resolution measurements.
"""
1. interpolate(L, alpha, LCP,toe=True) 
    which takes data: L in mm (depth ranges over which objects were visible) ,
    alpha in 1/mm (1/d where d is the dimension of the imaged object in the scan plane)
    and Low Contrast Penetration in mm.
    If toe=True then 'standard' rules apply for estimating crossing point of alpha axis:
        *Note that the case where a pipe was seen but could not measured is NOT accounted for
    If toe=False then a mimimum bound on R is calcuated by taking L straight down to 0
    Returns alpha, L(alpha)
    where alpha is a 200 point axis from 0 to the crossing point of L(alpha)
    and L(alpha) is a linearly interpolation function
                                                
2. res_int(alpha, linear) 
    which takes the output from (1), i.e. alpha and interpolated L(alpha).
    By performing a trapezoidal numerical integration, R is calculated.
    Depth of Field Lr and Characteristic Resolution Dr are calculated
    using an optimization function.
    Returns R, LR, DR.
"""

# Import standard packages for array manipulation, interpolation and optimization
import numpy as np
import sys
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

"""
Define the method prepare(L,alpha), 
which checks the data is well formatted and 
ordered such that L goes high to low and alpha
goes low to high.
"""

def check_array(x): # Check x is an array
    if type(x) is not np.ndarray:
        try: x = np.array(x)
        except:
            print('Problem Converting into array')
            sys.exit()
    return x
            
def check_floats(x): # Check the elements of array x are floats
    if x.dtype != 'float64': # Check elements are floats
        try: x = x.astype(float)
        except:
            print('L must contain only numbers')
            sys.exit()
    return x
            
def check(L,a): # Checks L and alpha are numerical arrays of the same length
    L = check_array(L)
    a = check_array(a)
    L = check_floats(L)
    a = check_floats(a)
    if L.size != a.size: # Check L and d have same number of elements
        print("L and alpha must be same size")
        sys.exit()
    return L,a

def sort_order(L,a):  # ensure L high to low, alpha low to high
    if L[0] < L[-1] : # if L is low to high, flip it
        L = np.flip(L)
    if a[0] > a[-1] : # if a is high to low, flip it
        a = np.flip(a)
    return L,a

def cut_zeros(L,a):             # Assumes L already sorted high to low
    zeros = np.where(L==0)[0]
    if zeros.size==0:           # All pipes may have been visible
        return L,a
    if zeros.size > 1:          # If there are multiple zeros
        L = L[:zeros[0]+1]        # Only keep the first one
        a = a[:zeros[0]+1]
    return L,a
        
def prepare(L,a): # Method to quickly combine above checks
    L,a = check(L,a)    # Make sure same length arrays
    L,a = sort_order(L,a) # Check L is high to low
    L,a = cut_zeros(L,a)  # Get rid of extra zeros
    return L,a

"""
Define the method interpolate(L,alpha).
Suitable crossing points of L and alpha axes are calculated.
"""

def alpha_axis_fix(L,alpha,toe): 
# Chooses most appropriate value for L=0, including/excluding 'toe' extrapolation
    if toe == False: # just set L=0 at same alpha as last point if no toe
        L = np.append(L,0)
        alpha = np.append(alpha,alpha[-1])
    elif L[-1] == 0:           # Case where a pipe was not seen
        straight_line = np.polyfit([alpha[-2],alpha[-3]],[L[-2],L[-3]],1)
        crossing = -straight_line[1]/straight_line[0]
        if crossing > alpha[-1]: 
            pass # Was biggest pipe not seen the likely limit?
        else: 
            alpha[-1] = crossing # If not, use linear interpolation for crossing point
    else:                          # Case where all pipes were seen
        straight_line = np.polyfit([alpha[-1],alpha[-2]],[L[-1],L[-2]],1)
        crossing = -straight_line[1]/straight_line[0]
        L, alpha = np.append(L,0), np.append(alpha,crossing) # Insert crossing poing by linear interp.
    return L, alpha

def interpolate(L,alpha,LCP,toe=True): # input is depth ranges L, corresponding alpha and low contrast penetration
    L, alpha = prepare(L,alpha)   # initial checks (see cell above)
    L, alpha = alpha_axis_fix(L,alpha,toe) # Determine crossing point of alpha axis
    L,alpha = np.insert(L,0,LCP), np.insert(alpha,0,0) # Insert crossing point of L axis
    linear = interp1d(alpha,L,kind='linear') # Join the dots interpolation
    return np.linspace(0,np.amax(alpha),200), linear  # Return 200 point alpha axis and L(alpha) as function

"""
Define res_int(alpha,linear)
Integrate under L(alpha) to get R.
Find a straight line that bisects area under L(alpha) to find Lr and Dr.
"""

def optimisation_fn(m,alpha,L,R): # m is the gradient of bisector
    difference = L-m*alpha # L(alpha) minus diagonal line
    crossing_index = np.where(difference < 0)[0][0] -1 # crossing point of lines
    area = np.trapz(difference[:crossing_index],alpha[:crossing_index]) # should be equal to 0.5 R
    return np.abs(0.5*R-area) 
    
def res_int(alpha,linear): # Calculate R, Lr, Dr from alpha axis and L as linearly interpolated function
    L = linear(alpha)
    R = np.trapz(L,alpha)
    grad = minimize_scalar(optimisation_fn,args=(alpha,L,R), method='Bounded',bounds=(0.001,1000)).x
    D_R = np.sqrt(grad/R)               # Characteristic Resolution
    L_R = np.sqrt(grad*R)               # Depth of Field
    return round(R,1), round(L_R,1), round(D_R,1)

