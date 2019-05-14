import pickle

with open('my_dataset.pickle', 'rb') as data:
    dataset = pickle.load(data)

print(dataset[660])
