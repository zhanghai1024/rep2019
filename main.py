
import mathfun.mathfunction as mf
#from math import *  # todo import funcitons
from classtesting import person, employee

import matplotlib.pyplot as plt


import numpy as np

 


def test1():

    x = [3, 6]

    for item in x:
        print('square of', item, 'is', mf.sqrt_of(item))
        print('power of ', item, 'is', mf.power_of(item))


def test2():
    name = ['Alan', 'Xia', 'Hai']
    
    age = [2, 32, 31]
    for index in range(len(name)):
    
        p = person(name[index], age[index])
        p.show()
        
    david=employee('David',31,99999)
    david.show()
    
    
#    print(david.__class__)
        

def main():

    test1()
    test2()
    # test3()
    
    
    #%%
if __name__ == '__main__':

    a = 2
    b = 3
    c = -5
    result = mf.demo(a, b, c)
    
 
    print(result)
    x=np.linspace(-5,5,500)
    y=a* np.power(x,2) + b* x +c

    plt.plot(x,y)
    
    plt.show()
    
    main()
