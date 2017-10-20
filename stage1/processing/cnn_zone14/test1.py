data = []
for i in range(98):
	data.append(i)

print(data)

chunks = len(data) // 5 + 1
for i in range(chunks):
     batch = data[i*5:(i+1)*5]
     print("batch - ",str(i),batch)