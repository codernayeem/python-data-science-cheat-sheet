# # # *************** Numpy *****************

# NumPy is used for working with arrays
# NumPy is short for "Numerical Python"
# It also has functions for working in domain of linear algebra, fourier transform, and matrices

# In Python we have lists that serve the purpose of arrays, but they are slow to process.
# NumPy aims to provide an array object that is up to 50x faster than traditional Python lists
# NumPy arrays are stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently
# NumPy is a Python library and is written partially in Python, but most of the parts that require fast computation are written in C or C++

import numpy as np

a = np.array(42) # 0-D Array
b = np.array([1, 2, 3, 4, 5]) # 1-D Array
c = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]) # 2-D Array
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]) # 3-D Array
print(type(b)) # <class 'numpy.ndarray'>

# data type
# i - integer
# b - boolean
# u - unsigned integer
# f - float
# c - complex float
# m - timedelta
# M - datetime
# O - object
# S - string
# U - unicode string
# V - fixed chunk of memory for other type ( void )

# numpy.bool_                     Boolean (True or False) stored as a byte
# numpy.byte                      Platform-defined
# numpy.ubyte                     Platform-defined
# numpy.short                     Platform-defined
# numpy.ushort                    Platform-defined
# numpy.intc                      Platform-defined
# numpy.uintc                     Platform-defined
# numpy.int_                      Platform-defined
# numpy.uint                      Platform-defined
# numpy.longlong                  Platform-defined
# numpy.ulonglong                 Platform-defined
# numpy.half / numpy.float16      Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
# numpy.single                    Platform-defined single precision float: typically sign bit, 8 bits exponent, 23 bits mantissa
# numpy.double                    Platform-defined double precision float: typically sign bit, 11 bits exponent, 52 bits mantissa
# numpy.longdouble                Platform-defined extended-precision float
# numpy.csingle                   Complex number, represented by two single-precision floats (real and imaginary components)
# numpy.cdouble                   Complex number, represented by two double-precision floats (real and imaginary components)
# numpy.clongdouble               Complex number, represented by two extended-precision floats (real and imaginary components)

# numpy.int8      Byte (-128 to 127)
# numpy.int16     Integer (-32768 to 32767)
# numpy.int32     Integer (-2147483648 to 2147483647)
# numpy.int64     Integer (-9223372036854775808 to 9223372036854775807)
# numpy.uint8     Unsigned integer (0 to 255)
# numpy.uint16    Unsigned integer (0 to 65535)
# numpy.uint32    Unsigned integer (0 to 4294967295)
# numpy.uint64    Unsigned integer (0 to 18446744073709551615)
# numpy.intp      Integer used for indexing, typically the same as ssize_t
# numpy.uintp     Integer large enough to hold a pointer
# numpy.float32
# numpy.float64 / numpy.float_         (builtin python float)
# numpy.complex64
# numpy.complex128 / numpy.complex_    (builtin python complex)

c = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=int) # integer
c = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype='i') # integer
c = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype='i4') # integer with 4 bytes
c = np.array([[1.5, 2, 3.5, 4, 5], [6, 0.4, 8, 9.1, 10]], dtype=np.float32) # folat with 32 bytes
c = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int16) # integer with 16 bytes (-32768 to 32767)

array1 = np.zeros((3, 4), dtype=int) # shape : (3, 4)
array2 = np.ones((3, 4), dtype=int) # shape : (3, 4)
array3 = np.full((3, 4), 5, dtype=int) # shape : (3, 4)
array43 = np.arange(1, 6) # [1, 2, 3, 4, 5]

array = np.array([[2, 5, 9], [5, 8, 3]]) # 2D array (metrics)
print('Array :', array)
print('Shape :', array.shape)
print('Dimension :', array.ndim)
print('Data Type :', array.dtype)

arr = np.array([1, 2, 3, 4], ndmin=5) # creating array of 5 Dimension


# ndarray supports indexing and negative indexing just like list
print(b[1])         # in list: lst[1]      1-D array
print(array[1, 2])  # in list: lst[1][2]   2-D array
print(array[0, -2]) # in list: lst[0][-2]  2-D array

# Array slicing
print(b[1:4]) # slicing 1-D array
print(b[1:4:2]) # slicing 1-D array with step
print(c[1, 1:4])   # slicing 2-D array (From the second element, slice elements from index 1 to index 3)
print(c[0:2, 2])   # slicing 2-D array (From both elements, return index 2)
print(c[0:2, 2:5]) # slicing 2-D array (From both elements, slice index 1 to index 4, returns a 2-D array)


# Copy vs View
array1 = np.array([[2, 5, 9], [5, 8, 3]], dtype=np.int8)

array2 = array1.copy()
array2[0, 1] = 10 # this will only change 'array2', not the 'array1'

array3 = array1.view()
array3[0, 1] = 10 # this will both change 'array1' and 'array3'

# Check copy / view
print('copy base : ', array2.base) # None
print('view base : ', array3.base) # <that array>


# Array Reshape
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(3, 4)     # reshaping to 2D array (3 * 4 = 12 = total elements)
newarr = arr.reshape(2, 3, 2)  # reshaping to 3D array (2 * 3 * 2 = 12 = total elements)
# will raise an error, if it can not be reshaped to the given shape


# Iterating
arr = np.array([[3, 2, 4], [5, 0, 1]])
for x in arr: # 2D array
  for y in x:
    print(y)

# Iterating n-D array using nditer()
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
for x in np.nditer(arr):
  print(x)

# Enumerated Iteration Using ndenumerate()
for idx, x in np.ndenumerate(arr):
  print('Index :', idx, 'Value :', x)



# Joining arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))  # [1 2 3 4 5 6]

arr3 = np.array([[1, 2], [3, 4]])
arr4 = np.array([[5, 6], [7, 8]])
arr = np.concatenate((arr3, arr4), axis=1) # along rows     | [[1 2 5 6], [3 4 7 8]]
arr = np.concatenate((arr3, arr4), axis=0) # along columns  | [[1 2], [3 4], [5 6], [7 8]]

# same as concatenation, the only difference is that stacking is done along a new axis
arr = np.stack((arr3, arr4), axis=1)  # [[1 4], [2 5], [3 6]]
arr = np.stack((arr3, arr4), axis=0)  # [[1 2 3], [4 5 6]]
# Stacking Along Rows, Columns and Height (depth)
arr = np.hstack((arr3, arr4))  # [1 2 3 4 5 6]
arr = np.vstack((arr3, arr4))  # [[1 2 3], [4 5 6]]
arr = np.dstack((arr1, arr2))  # [[[1 4], [2 5], [3 6]]]



# Spliting arrays (reverse operation of Joining)
arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)  # [array([1, 2]), array([3, 4]), array([5, 6])]
# If the array has less elements than required, it will adjust from the end accordingly
newarr = np.array_split(arr, 4)  # [array([1, 2]), array([3, 4]), array([5]), array([6])]

# Split the 2-D array into three 2-D arrays
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
newarr = np.array_split(arr, 3)  # [array([[1, 2], [3, 4]]), array([[5, 6], [7, 8]]), array([[ 9, 10], [11, 12]])]

# Spliting along rows
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3, axis=1) # [array([[ 1], [ 4], [ 7], [10], [13], [16]]), array([[ 2], [ 5], [ 8], [11], [14], [17]]), array([[ 3], [ 6], [ 9], [12], [15], [18]])]
newarr = np.hsplit(arr, 3)              # same
# Spliting along culmns
newarr = np.array_split(arr, 3, axis=0) # [array([[1, 2, 3], [4, 5, 6]]), array([[ 7,  8,  9], [10, 11, 12]]), array([[13, 14, 15], [16, 17, 18]])]
newarr = np.vsplit(arr, 3)              # same
# Spliting along Height (depth)
# newarr = np.dsplit(arr, 3) # height/depth split only works for 3 or higher dimension



# Searching Arrays
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)    # Find the indexes where the value is 4
x = np.where(arr%2 == 0)  # Find the indexes where the values are even

# Search Sorted (returns the index where the specified value would be inserted to maintain the search order)
arr = np.array([6, 7, 8, 9]) # a sorted arrary
x = np.searchsorted(arr, 7) # Find the indexes where the value 7 should be inserted : 2
x = np.searchsorted(arr, 7, side='right') # Find the indexes where the value 7 should be inserted, starting from the right : 2
x = np.searchsorted(arr, [4, 2, 6]) # Find the indexes where the values 2, 4, and 6 should be inserted : [2, 1, 3]



# Sorting
arr = np.array([3, 2, 0, 1])
newarr = np.sort(arr) # return the sorted array : [0, 1, 2, 3]

arr = np.array([[3, 2, 4], [5, 0, 1]])
print(np.sort(arr))  # [[2, 3, 4], [0, 1, 5]]



# Filter
arr = np.array([41, 42, 43, 44])
x = [True, False, True, False]
newarr = arr[x] # [41, 43]

filter_arr = arr > 42 # [False, False, True, True]
newarr = arr[filter_arr] # [43, 44]
# in one line : newarr = arr[arr > 42]

filter_arr = arr % 2 == 0 # [False, True, False, True]
newarr = arr[filter_arr] # [42, 44]

arr = np.array([[3, 2, 4], [5, 0, 1]])
newarr = arr[arr % 3 != 1] # [3 2 5 0]



# Random
from numpy import random
arr = random.randint(100, size=(5)) # Generate a 1-D array containing 5 random integers from 0 to 100
arr = random.randint(100, size=(3, 5)) # Generate a 2-D array with 3 rows, each row containing 5 random integers from 0 to 100
arr = random.rand(3, 5) # Generate a 2-D array with 3 rows, each row containing 5 random numbers
arr = random.choice([3, 5, 7, 9], size=(3, 5))

# Generate a 1-D array containing 50 values, where each value has to be 3, 5, 7 or 9.
# The probability for the value to be 3 is set to be 0.1
# The probability for the value to be 5 is set to be 0.3
# The probability for the value to be 7 is set to be 0.6
# The probability for the value to be 9 is set to be 0
# The sum of all probability numbers should be 1
arr = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(50))

arr = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(3, 5)) # 2D array

arr = np.array([1, 2, 3, 4, 5])
random.shuffle(arr) # shuffle changes orginial array
newarr = random.permutation(arr)




# Numpy ufunc
# ufuncs stands for "Universal Functions" and they are NumPy functions that operates on the ndarray object
# ufuncs are used to implement vectorization in NumPy which is way faster than iterating over elements

# Create Your Own ufunc
# The frompyfunc() method takes the following arguments:
#   function - the name of the function.
#   inputs   - the number of input arguments (arrays).
#   outputs  - the number of output arrays.

def mycalc(x, y):
  return (x + y) / 2

mycalc = np.frompyfunc(mycalc, 2, 1) # here, 2 is number of input arrays, 1 is number of output arrays

print(mycalc([1, 2, 3, 4], [5, 6, 7, 8])) # [3.0 4.0 5.0 6.0]

print(type(np.add)) # <class 'numpy.ufunc'>
print(type(mycalc))  # <class 'numpy.ufunc'>


# ufunc Simple Arithmetic
arr1 = np.array([10, 11, 12, 13, 14, 15])
arr2 = np.array([20, 21, 22, 23, 24, 25])

newarr = np.add(arr1, arr2) # [30 32 34 36 38 40]
newarr = arr1 + arr2
newarr = np.subtract(arr1, arr2) # [-10 -10 -10 -10 -10 -10]
newarr = arr1 - arr2
newarr = np.multiply(arr1, arr2) # [200 231 264 299 336 375]
newarr = arr1 * arr2
newarr = np.divide(arr1, arr2) # [0.5 0.52380952 0.54545455 0.56521739 0.58333333 0.6] 
newarr = arr1 / arr2
newarr = np.power(arr1, arr2) # [ 7766279631452241920  3105570700629903195  5729018530666381312 -4649523274362944347 -1849127232522420224  1824414961309619599] 
newarr = arr1 ** arr2
newarr = np.mod(arr1, arr2) # [10 11 12 13 14 15]
newarr = arr1 % arr2
newarr = np.divmod(arr1, arr2) # (with remainders) (array([0, 0, 0, 0, 0, 0]), array([10, 11, 12, 13, 14, 15])) 
newarr = arr1 // arr2 # [0, 0, 0, 0, 0, 0]

newarr = np.absolute([-1, -2, 1, 2, 3, -4]) # [1, 2, 1, 2, 3, 4]

# just like that we can give any other value, like:
newarr = arr1 + 2 # [12 13 14 15 16 17] 
newarr = arr1 % 3 # [1 2 0 1 2 0]



# ufunc mathematical operations
arr = np.array([-3.1666, 3.6667])

# Remove the decimals, and return the float number closest to zero
newarr = np.trunc(arr)      # [-3.  3.]
newarr = np.fix(arr)        # [-3.  3.]
# round float
newarr = np.around(arr)  # [-3.  4.]
# round float to 2 decimal points
newarr = np.around(arr, 2)  # [-3.17  3.67]
# The floor() function rounds off decimal to nearest lower integer
newarr = np.floor(arr)      # [-4.  3.]
# The ceil() function rounds off decimal to nearest upper integer
newarr = np.ceil(arr)       # [-3.  4.]

arr = np.arange(1, 10)

newarr = np.log2(arr)  # base 2
newarr = np.log10(arr) # base 10
newarr = np.log(arr)   # base e

from math import log
nplog = np.frompyfunc(log, 2, 1)
# np do not provide func for log with any base, so we create our own
newarr = nplog(arr, 4)   # any base

# like these, there is:
# np.sin(), np.cos(), np.tan(), np.arcsin(), np.arccos(), np.arctan()
# np.sinh(), np.cosh(), np.tanh(), np.arcsinh(), np.arccosh(), np.arctanh()
# np.deg2rad(), np.rad2deg(), np.hypot()

arr = np.array([1, 2, 3, 4])
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])

# Summations
newarr = np.sum([arr1, arr2]) # 12
newarr = np.sum([arr1, arr2], axis=1) # [6 6]

# Cummulative sum : partially adding the elements in array.
# E.g. The partial sum of [1, 2, 3, 4] would be [1, 1+2, 1+2+3, 1+2+3+4] = [1, 3, 6, 10]
newarr = np.cumsum(arr) # [1, 3, 6, 10]

# Products
newarr = np.prod(arr) # 24 because 1*2*3*4 = 24
newarr = np.prod([arr1, arr2], axis=1) # [6 6]

# Just like Cummulative sum, there is Cummulative Product
# E.g. The partial sum of [1, 2, 3, 4] would be [1, 1*2, 1*2*3, 1*2*3*4] = [1, 2, 6, 24]
newarr = np.cumprod(arr) # [1, 2, 6, 24]

# Discrete Difference
arr = np.array([10, 15, 25, 5])
newarr = np.diff(arr) # [5 10 -20] because 15-10=5, 25-15=10, and 5-25=-20

# LCD & GCD
lcd = np.lcm.reduce(arr) # 150
gcd = np.gcd.reduce(arr) # 5


np.minimum([2, 3, 4], [1, 5, 2]) # array([1, 3, 2])
np.maximum([2, 3, 4], [1, 5, 2]) # array([2, 5, 4])

min_value = arr.min()
max_value = arr.max()
mean_value = arr.mean()
print(arr.T)              # transpose


# Sets in Numpy (A set in mathematics is a collection of unique elements)
arr = np.array([4, 6, 8, 4, 6])
newarr = np.unique(arr)

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

newarr = np.union1d(arr1, arr2)                         # Union
newarr = np.intersect1d(arr1, arr2, assume_unique=True) # Intersection
newarr = np.setdiff1d(arr1, arr2, assume_unique=True)   # Difference
newarr = np.setxor1d(arr1, arr2, assume_unique=True)    # Symmetric Difference

# assume_unique : which if set to True can speed up computation. It should always be set to True when dealing with sets

