from pdr.pdr import BasePDR, AHRSPDR, Location

pdr = AHRSPDR(Location(4, 1))
loc1 = pdr.getCurrentLocation(1000, [1.0, 2.0, 3.0], [4.5, 5.3, 2.1], [1.1, 4.9, 5.3])
loc2 = pdr.getCurrentLocation(2000, [1.0, 2.0, 8.0], [4.5, 5.3, 2.1], [1.1, 4.9, 5.3])
loc3 = pdr.getCurrentLocation(3000, [1.0, 2.0, 16.0], [4.5, 5.3, 2.1], [1.1, 4.9, 5.3])
loc4 = pdr.getCurrentLocation(4000, [1.0, 2.0, 8.0], [4.5, 5.3, 2.1], [1.1, 4.9, 5.3])
loc5 = pdr.getCurrentLocation(5000, [1.0, 2.0, 2.0], [4.5, 5.3, 2.1], [1.1, 4.9, 5.3])

print("(" + str(loc1.getX()) + ", " + str(loc1.getY()) + ")")
print("(" + str(loc2.getX()) + ", " + str(loc2.getY()) + ")")
print("(" + str(loc3.getX()) + ", " + str(loc3.getY()) + ")")
print("(" + str(loc4.getX()) + ", " + str(loc4.getY()) + ")")
print("(" + str(loc5.getX()) + ", " + str(loc5.getY()) + ")")
