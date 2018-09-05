#to open or de-pickle the file:
import pickle as pickle

infile = open(agents_dict, 'rb')
new_dict = pickle.load(infile)
infile.close()

print(new_dict)
print(new_dict==agents_dict)
print(type(new_dict))