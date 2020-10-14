import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(nrows=1, ncols=3, sharey=False)
fig.set_size_inches(10, 3)

for i, theta in enumerate([1, 10, 30]):
    with open(f"results/plots/SART_reconstruction_step_{theta}.txt", mode="r", encoding="utf8") as f:
        vals = []
        x = np.arange(1, 10, 1)
        for s in f.readlines()[1:10]:
            vals.append(float(s.split("RME")[-1].strip("\n\t\r")))
        axs[i].plot(x, vals)
        axs[i].set_title(str(theta))
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel("RME")
plt.show()
