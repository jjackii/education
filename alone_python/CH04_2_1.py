numbers = [1, 3, 21, 5, 12, 1, 8, 2, 21, 7, 2, 5, 2, 2, 8, 6, 7, 3]
numbers.sort()
# print(numbers)
counter = {}

for num in numbers:
    if num not in counter:
        counter[num] = 1
    else:
        counter[num] = counter[num] + 1

print(counter)