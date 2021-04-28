import time, csv, random
from itertools import count

x = [0]
t1 = [0]
t2 = [0]

index = count()

while True:
    with open("data\\changing_data.csv", "w") as file:
        writer = csv.writer(file)
        for xx, i, ii in zip(x, t1, t2):
            writer.writerow([xx, i, ii])
                
    x.append(next(index))
    t1.append(t1[-1] + random.randint(-4, 10))
    t2.append(t2[-1] + random.randint(-4, 10))
    time.sleep(1)
