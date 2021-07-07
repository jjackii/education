# No.2164
## sol(1). timeout
from collections import deque

q = deque()
n = int(input())
tmp=0

for i in range(1, n+1):
    q.append(i)

while 1:  
    if len(q) == 1:
        break

    # 제일 위의 카드 버리기
    q.popleft()

    # 제일 위의 카드를 맨밑으로 옮기기
    tmp = q.popleft()
    q.append(tmp)
    
print(q.pop())

## sol(2). 
from collections import deque

q = deque()
n = int(input())

for i in range(1, n+1):
    q.append(i)

while 1:
    if len(q) == 1:
        break

    q.popleft()
    q.rotate(-1)

print(q.pop())