#exercise 3
import csv
import numpy as np
import matplotlib.pyplot as plt

csv_file_path = 'data_file.csv'

with open(csv_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["precision", "recall"])
    writer.writerows([[0.013,0.951],
                      [0.376,0.851],
                      [0.441,0.839],
                      [0.570,0.758],
                      [0.635,0.674],
                      [0.721,0.604],
                      [0.837,0.531],
                      [0.860,0.453],
                      [0.962,0.348],
                      [0.982,0.273],
                      [1.0,0.0]])

results = []
with open(csv_file_path) as result_csv:
    csv_reader = csv.reader(result_csv, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
        results.append([float(i) for i in row])
    results = np.stack(results)

plt.plot(results[:, 1], results[:, 0])
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()