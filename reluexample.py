import numpy as np

inputs =[0, 2, -1, 3.3, -2.7, 1.1, 2.2, 100]

output =[]
#for i in inputs:
 #   if i > 0:
  #      output.append(i**2)
   # else:
    #    output.append(0)

#for i in inputs:
 #   output.append(max(0, i)) # prints the larger value

output = np.maximum(0, inputs)

print(output)