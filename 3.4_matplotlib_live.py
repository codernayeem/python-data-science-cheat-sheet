from matplotlib import pyplot as plt
import pandas as pd
import random
from itertools import count
from matplotlib.animation import FuncAnimation
plt.style.use('bmh')

# index = count()
# x = []
# y = []

# def animate(i):
#     x.append(next(index))
#     y.append(random.randint(1, 10))

#     plt.cla()
#     plt.plot(x, y)

#     plt.title('Lavel Monitor')
#     plt.xlabel('Count')
#     plt.ylabel('Levels')
#     plt.yticks(ticks=range(12))


# ani = FuncAnimation(plt.gcf(), animate, interval=1000)

# plt.tight_layout()
# plt.show()


def animate(i):
    df = pd.read_csv('data\\changing_data.csv')

    x = df.iloc[-50:, 0]
    y1 = df.iloc[-50:, 1]
    y2 = df.iloc[-50:, 2]

    plt.cla() # clear axis
    plt.plot(x, y1, label='Ajaira LTD')
    plt.plot(x, y2, label='Tawhid Afridi')

    plt.fill_between(x, y1, y2, where=y1 > y2, color='b', alpha=0.5, interpolate=True)
    plt.fill_between(x, y1, y2, where=y1 <= y2, color='r', alpha=0.5, interpolate=True)

    plt.title('Channel Subscriptions')
    plt.xlabel('Days')
    plt.ylabel('Subscriptions')
    plt.legend()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()
