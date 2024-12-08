
import pandas as pd

col = ["零件品种", "工艺数目", "小车数量", "零件数量", "优势算法"]
DF = pd.DataFrame(columns = col)

for kinds in [1, 5, 10]:
	for processes in [5, 10]:
		df = pd.read_csv("Data_{}_{}.csv".format(kinds, processes), header = None)
		for agvs, r1 in zip([2, 5, 10], [0, 6, 12]):
			h1 = df.iloc[r1][[2, 4, 6, 8]]
			e1 = df.iloc[r1][[3, 5, 7, 9]]
			h1.index = list(range(4))
			e1.index = list(range(4))
			th1, te1 = h1[0], e1[0]
			for parts, r2 in zip([20, 40, 60, 80, 100], range(r1+1, r1+6)):
				h = df.iloc[r2][[2, 4, 6, 8]]
				s = df.iloc[r2][[3, 5, 7, 9]]
				h.index = list(range(4))
				s.index = list(range(4))
				h1[0] = parts//20 * th1
				e1[0] = parts//20 * te1
				ave = (h + s + h1 + e1) / 4
				efc, efa = float("inf"), ""
				for a, alg in zip([s, h, h1, e1], ["随机调度", "就近预订", "就近预订（批）", "EDA（批）"]):
					efc1 = a[0]/ave[0] + a[1]/ave[1] - 0.5*a[2]/ave[2] - 0.5*a[3]/ave[3]
					if efc1 < efc: efc, efa = efc1, alg
				data = pd.DataFrame([[kinds, processes, agvs, parts, efa]], columns = col)
				DF = DF.append(data)

print(DF)
DF.to_csv("Statistic.csv", index = False)

