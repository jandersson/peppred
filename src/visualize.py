import matplotlib.pyplot as plt
import seaborn as sns
from data import get_data, transform_data
data = transform_data(get_data())
instances = [item['sequence'] for item in data]
sizes = [len(rec) for rec in instances]
plt.hist(sizes, bins=30)
plt.title(f"{len(sizes)} sequences\nLengths {min(sizes)} to {max(sizes)}")
plt.xlabel("Sequence length")
plt.ylabel("Count")
plt.savefig('../report/img/sequence-hist.png')
plt.show()
 