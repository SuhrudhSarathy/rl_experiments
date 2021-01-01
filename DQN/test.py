import random
memory = []
capacity = 100
position = 0
def push(a):
    global memory
    global capacity
    global position
    if len(memory) < capacity:
        memory.append(None)
    memory[position] = a
    position = (position + 1) % capacity
    print(memory)
for i in range(100):
    push(i)
print(random.sample(memory,50 ))

