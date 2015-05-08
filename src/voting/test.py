import matplotlib.pyplot as plt
import numpy as np
import asyncio

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], '-o')
ax.set_autoscaley_on(True)
ax.relim()
ax.grid()


@asyncio.coroutine
def update_line():
    Z = 0
    while True:
        line.set_xdata(np.append(line.get_xdata(), Z))
        line.set_ydata(np.append(line.get_ydata(), np.sin(0.2*Z)))
        Z += 1
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        yield from asyncio.sleep(0.5)

loop = asyncio.get_event_loop()
loop.run_until_complete(update_line())
#loop.close()

