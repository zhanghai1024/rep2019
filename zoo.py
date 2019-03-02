# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 22:19:30 2019

@author: hzhang7
"""
import json
class Animal():
    def __init__(self,name,species):
        self.name =name
        self.species = species
        self.stomach =[]   # an empty list
        
    def eat(self,food):
        self.stomach.append(food)
        
    def __call__(self,food):
        self.eat(food)    # __call__ method   
        
    def __repr__(self):
        return "Animal ({},{})".format(self.name,self.species)  # format method
    
class Zoo():
    def __init__(self):
        self.animals = []    # a list of animal objects
        
    def hire(self, zk):
        self.zk = zk       # add a new self attribute which is an Zookeeper object
        
    def feed_animals(self):
        
        if 'zk' not in self.__dict__:
            raise ValueError("No zookeeper")   # raise exceptions 
            
        for animal in self.animals :
            self.zk.feed_animal(animal)   # call the Zookeeper's method
            
    def add_animal(self,animal):
        self.animals.append(animal)   # append add animal to the list
        
    def __iter__(self):
        return iter(self.animals)  # iterable 

class Zookeeper:

    def __init__(self, name, ssn) :
        self.name = name
        self.ssn = ssn
        self.foods = self._getfoods()   #private funciton to get initilizaiton of attribute
        
    def _getfoods(self) :
        with open("what_they_eat.json",'r') as f:   # open jason file
            data= f.read() 
        return json.loads(data)
    
    def feed_animal(self, animal) :
        pref_food = self.foods.get(animal.species,'pellet')  # get method......
        animal(pref_food)   # call method

def main():
       
    zoo= Zoo()
    
    zoo.add_animal(Animal("Rover", "lion"))
    zoo.add_animal(Animal("Fido", "tiger"))
    zoo.add_animal(Animal("Pep","dog"))
    print(zoo.animals[1])
    jill = Zookeeper("Jill Mac", "222-3445")
    
    zoo.hire(jill)
    zoo.feed_animals()
    
    
    
    for animal in zoo :
        print("Animal {name} the {species} ate {stomach}".format(**animal.__dict__))  # print...
    
if __name__ =="__main__":
    
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        