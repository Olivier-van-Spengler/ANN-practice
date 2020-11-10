from scipy.io import loadmat


def load_mat(file_name):
    annots = loadmat(file_name)
    return annots


dataset = load_mat('matlab/dataset.mat')

# Check that dataset is a tuple
print('dataset has type:', type(dataset))

# Print the number of elements in dataset
print('dataset has {:,} elements '.format(len(dataset)))

print(dataset)

