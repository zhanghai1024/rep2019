
class person(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def show(self):
        print(self.name, 'is', self.age, 'years old')


class employee(person):

    def __init__(self,name, age, salary):
        super(employee,self).__init__(name, age)        
        self.salary = salary
        
    def show(self):

        print(self.name, 'is', self.age, 'years old', 'and salary is',self.salary)
