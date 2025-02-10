class shape:
  def __init__(self, color,is_filled):
    self.color = color
    self.is_filled = is_filled
    
  def describe(self):
    print(f"it is {self.color} and {'filled' if self.is_filled else 'Not filled'}")  
    
class Circle (shape):
  def __init__(self,color, is_filled,radius):
    super(). __init__ (color, is_filled) 
    self.radius = radius
    
    
circle = Circle (color= 'red', is_filled= True, radius = 5)  
circle. describe()

 

