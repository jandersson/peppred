import matplotlib.pyplot as plt
import seaborn as sns
sizes = [len(rec) for rec in instances]
plt.hist(sizes, bins=30)
plt.title(f"{len(sizes)} sequences\nLengths {min(sizes)} to {max(sizes)}")
plt.xlabel("Sequence length")
plt.ylabel("Count")
plt.savefig('../report/img/sequence-hist.png')
plt.show()
 