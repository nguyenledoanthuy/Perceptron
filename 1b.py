import matplotlib.pyplot as plt

# Decoration the graph 
plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
# Set the maximum values of cordinate 
plt.xlim(-1.5, 2.5), plt.ylim(-1.5,2.5)
# Drawing the line through [x1,y1], [x2,y2] points with the label 
plt.plot([-2,1],[2.5,-1], label = "0.4x1 + 0.9x2 - 0.1 = 0", linewidth=0.5)
# Add the three points to the cordinate which have class = +1 
plt.scatter([1,0,1],[1,1,0])
plt.scatter([0],[0])
plt.show()