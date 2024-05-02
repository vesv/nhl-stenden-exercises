#exercise 2
import numpy as np

coords = np.array([[10, 5, 15, 6, 0],
                    [11, 3, 13, 6, 0],
                    [5, 3, 13, 6, 1],
                    [4, 4, 13, 6, 1],
                    [6, 5, 13, 16, 1]])

flipped_coords = np.flip(np.flipud(coords))
flipped_coords = np.flipud(flipped_coords)
print(flipped_coords)

#the first issue is that numpy was imported after it was defined
#after that i used numpy to first flip the order of items in the individal lists in the array
#then i flip the order of the lists in the array
#print the flipped array