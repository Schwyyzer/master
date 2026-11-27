import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Replace these with your real data -- all three must share this x.
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = x ** 2

fig, ax_left = plt.subplots(figsize=(8, 5))

ax_left.plot(x, y1, color="tab:blue", label="y1")
ax_left.plot(x, y2, color="tab:green", label="y2")
ax_left.set_xlabel("x")
ax_left.set_ylabel("y1 / y2 (left axis)")

ax_right = ax_left.twinx()
ax_right.plot(x, y3, color="tab:red", label="y3")
ax_right.set_ylabel("y3 (right axis)")

# twinx() makes a second Axes, so legends are separate by default --
# collect both and combine into one.
lines_left, labels_left = ax_left.get_legend_handles_labels()
lines_right, labels_right = ax_right.get_legend_handles_labels()
ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")

fig.tight_layout()
fig.savefig("plot.png", dpi=200)
print("saved plot.png")
