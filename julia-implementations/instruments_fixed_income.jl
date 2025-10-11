using Plots



# Visualizing the bond


t = 1:0.5:15

FV = 100

c = 0.05 

C = c/2 * FV

# Let's assume a simple constant yield curve

r = 0.05

class TermStructure

