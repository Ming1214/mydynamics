
import pandas as pd
from matplotlib import pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  #解决保存图像是负号'-'显示为方块的问题

DF = pd.read_csv("Statistic.csv")

for i, p, kinds, processes in zip(range(6),
		[1, 4, 2, 5, 3, 6], [1, 1, 5, 5, 10, 10], [5, 10, 5, 10, 5, 10]):
	plt.subplot(2, 3, p)
	p1, p2, p3, p4 = None, None, None, None
	for j in range(15):
		k = i * 15 + j
		y, x, c = DF.iloc[k]["小车数量"], DF.iloc[k]["零件数量"], DF.iloc[k]["优势算法"]
		if c == "EDA（批）" or c == "EDA（批）*":
			c, l1 = "red", "EDA（批）"
			p1, = plt.plot([x], [y], color = c, marker = "o", label = l1)
		if c == "就近预订（批）" or c == "就近预订（批）*":
			c, l2 = "blue", "就近预订（批）"
			p2, = plt.plot([x], [y], color = c, marker = "o", label = l2)
		if c == "就近预订" or c == "就近预订*":
			c, l3 = "gray", "就近预订"
			p3, = plt.plot([x], [y], color = c, marker = "o", label = l3)
		if c == "随机调度" or c == "随机调度*":
			c, l4 = "black", "随机调度"
			p4, = plt.plot([x], [y], color = c, marker = "o", label = l4)
	plt.legend(handles = [pi for pi in [p1, p2, p3, p4] if pi], loc = "upper right")
	plt.title("零件品种{}；工艺数目{}".format(kinds, processes))
	plt.yticks([2, 5, 10], [2, 5, 10])
	plt.ylabel("小车数量")
	plt.xticks([20, 40, 60, 80, 100], [20, 40, 60, 80, 100])
	plt.xlabel("零件数量")
	plt.grid()
plt.suptitle("     算法的优势分布", fontsize = 14)
plt.tight_layout(pad = 2, h_pad = 0, w_pad = -2)
plt.show()





