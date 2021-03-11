from pdr.pdr import BasePDR, AHRSPDR, Location

pdr = BasePDR(Location(4, 1))
pdr.getCurrentLocation(2232, [[3567, 1.0, 2.0, 3.0], [4232, 4.5, 5.3, 2.1], [3423, 1.1, 4.9, 5.3]], 3, 1)
