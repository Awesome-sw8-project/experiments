from pdr.pdr import BasePDR, AHRSPDR, Location

pdr = AHRSPDR(Location(208.86206, 216.74796))
loc1 = pdr.getCurrentLocation(1000, [1.0, 2.0, 5.4494324], [4.5, 5.3, 2.1], [0.4945526, 0.122528076, 0.25663757])
loc2 = pdr.getCurrentLocation(2000, [1.0, 2.0, 5.6469574], [4.5, 5.3, 2.1], [0.10040283, 0.087371826, 0.27635193])
loc3 = pdr.getCurrentLocation(3000, [1.0, 2.0, 6.6172028], [4.5, 5.3, 2.1], [-0.28788757, 0.07672119, 0.31736755])
loc4 = pdr.getCurrentLocation(4000, [1.0, 2.0, 7.190613], [4.5, 5.3, 2.1], [-0.53715515, 0.0655365, 0.35305786])
loc5 = pdr.getCurrentLocation(5000, [1.0, 2.0, 7.6119995], [4.5, 5.3, 2.1], [-0.6633911, 0.10974121, 0.31950378])
loc6 = pdr.getCurrentLocation(6000, [1.0, 2.0, 8.436798], [4.5, 5.3, 2.1], [-0.5541992, 0.28659058, 0.17089844])
loc7 = pdr.getCurrentLocation(7000, [1.0, 2.0, 9.450134], [4.5, 5.3, 2.1], [-1.0628662, -0.043655396, -0.07626343])
loc8 = pdr.getCurrentLocation(8000, [1.0, 2.0, 13.235367], [4.5, 5.3, 2.1], [-1.1534119, -0.044189453, 0.02229309])
loc9 = pdr.getCurrentLocation(9000, [1.0, 2.0, 14.442062], [4.5, 5.3, 2.1], [-0.7986908, 0.12466431, -0.008071899])
loc10 = pdr.getCurrentLocation(10000, [1.0, 2.0, 16.289185], [4.5, 5.3, 2.1], [-0.59840393, 0.41122437, 0.020690918])
loc11 = pdr.getCurrentLocation(11000, [1.0, 2.0, 20.341965], [4.5, 5.3, 2.1], [-0.32037354, 0.24822998, 0.056381226])
loc12 = pdr.getCurrentLocation(12000, [1.0, 2.0, 17.854996], [4.5, 5.3, 2.1], [0.40454102, -0.35151672, 0.007904053])
loc13 = pdr.getCurrentLocation(13000, [1.0, 2.0, 13.738754], [4.5, 5.3, 2.1], [0.9494171, -0.39466858, -0.06347656])
loc14 = pdr.getCurrentLocation(14000, [1.0, 2.0, 11.000381], [4.5, 5.3, 2.1], [1.2375793, -0.40478516, -0.11140442])
loc15 = pdr.getCurrentLocation(15000, [1.0, 2.0, 8.473312], [4.5, 5.3, 2.1], [1.4160004, -0.27694702, -0.0975647])
loc16 = pdr.getCurrentLocation(16000, [1.0, 2.0, 7.112808], [4.5, 5.3, 2.1], [1.3478241, -0.090530396, -0.10021973])
loc17 = pdr.getCurrentLocation(17000, [1.0, 2.0, 7.1505127], [4.5, 5.3, 2.1], [1.1811218, -0.020751953, -0.07092285])
loc18 = pdr.getCurrentLocation(18000, [1.0, 2.0, 6.7470856], [4.5, 5.3, 2.1], [1.0239868, -0.034591675, -0.05015564])
loc19 = pdr.getCurrentLocation(19000, [1.0, 2.0, 6.5657196], [4.5, 5.3, 2.1], [0.77311707, -0.047927856, -0.048568726])
loc20 = pdr.getCurrentLocation(19000, [1.0, 2.0, 6.6429443], [4.5, 5.3, 2.1], [0.49508667, -0.004776001, -0.08157349])
loc21 = pdr.getCurrentLocation(19000, [1.0, 2.0, 6.90271], [4.5, 5.3, 2.1], [0.25379944, 0.029327393, -0.121536255])
loc22 = pdr.getCurrentLocation(19000, [1.0, 2.0, 7.160095], [4.5, 5.3, 2.1], [0.06098938, -0.0026397705, -0.14390564])

print("(" + str(loc1.getX()) + ", " + str(loc1.getY()) + ")")
print("(" + str(loc2.getX()) + ", " + str(loc2.getY()) + ")")
print("(" + str(loc3.getX()) + ", " + str(loc3.getY()) + ")")
print("(" + str(loc4.getX()) + ", " + str(loc4.getY()) + ")")
print("(" + str(loc5.getX()) + ", " + str(loc5.getY()) + ")")
print("(" + str(loc6.getX()) + ", " + str(loc6.getY()) + ")")
print("(" + str(loc7.getX()) + ", " + str(loc7.getY()) + ")")
print("(" + str(loc8.getX()) + ", " + str(loc8.getY()) + ")")
print("(" + str(loc9.getX()) + ", " + str(loc9.getY()) + ")")
print("(" + str(loc10.getX()) + ", " + str(loc10.getY()) + ")")
print("(" + str(loc11.getX()) + ", " + str(loc11.getY()) + ")")
print("(" + str(loc12.getX()) + ", " + str(loc12.getY()) + ")")
print("(" + str(loc13.getX()) + ", " + str(loc13.getY()) + ")")
print("(" + str(loc14.getX()) + ", " + str(loc14.getY()) + ")")
print("(" + str(loc15.getX()) + ", " + str(loc15.getY()) + ")")
print("(" + str(loc16.getX()) + ", " + str(loc16.getY()) + ")")
print("(" + str(loc17.getX()) + ", " + str(loc17.getY()) + ")")
print("(" + str(loc18.getX()) + ", " + str(loc18.getY()) + ")")
print("(" + str(loc19.getX()) + ", " + str(loc19.getY()) + ")")
print("(" + str(loc20.getX()) + ", " + str(loc20.getY()) + ")")
print("(" + str(loc21.getX()) + ", " + str(loc21.getY()) + ")")
print("(" + str(loc22.getX()) + ", " + str(loc22.getY()) + ")")
