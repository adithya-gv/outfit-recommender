f = open("data/data.txt", "r")

sum = 0.0
count = 0
for x in f:
    sum = sum + float(x)
    count = count + 1

print(sum / count)