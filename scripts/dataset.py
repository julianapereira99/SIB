from si.util import StandardScaler
from si.data import Dataset
import os

DIR = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(DIR, '../datasets/lr-example1.data')

print("Without labels\n\n")

dataset = Dataset.from_data(filename, labeled=False)

print("X=\n", dataset.X[:10, :])
print("Y=\n", dataset.Y)
print("has label", dataset.hasLabel())
print("Number of features", dataset.getNumFeatures())
print("Number of classes", dataset.getNumClasses())

print("\nScalling")
sc = StandardScaler()
ds2 = sc.fit_transform(dataset)
print("X=\n", ds2.X[:10, :])

print("\n\nWith labels\n\n")

dataset = Dataset.from_data(filename, labeled=True)
print("X=\n", dataset.X[:10, :])
print("Y=\n", dataset.Y[:10])
print("has label", dataset.hasLabel())
print("Number of features", dataset.getNumFeatures())
# Those are not classes... Y is a continuous variable
print("Number of classes", dataset.getNumClasses())
