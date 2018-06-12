import word2vec
model = word2vec.load("vectors.bin")
print(model.vectors)