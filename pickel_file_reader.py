import pickle

with open('/home/akino/Desktop/RESUME/cache/resume_texts.pkl', 'rb') as f:
    data = pickle.load(f)

print(data[0][0])
print("\n\n")
print(data[1][0])
print("\n\n")
