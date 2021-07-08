# No.13305
## sol(1).
n = int(input())
dist = list(map(int, input().split()))
cost = list(map(int, input().split()))

result=dist[0]*cost[0]
temp=cost[0]

for i in range(1, n-1):
    if cost[i] < temp:
        temp = cost[i]

    result += dist[i] * temp
print(result)
# TypeError: 'int' object is not subscriptable
# 정수인덱스로 접근하는 것이 불가능 할 때


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


# No.1021
## sol(1).
from collections import deque

n, m = map(int, input().split())
idx = list(map(int, input().split()))
q = deque()
count=0

for i in range(1, n+1):
    q.append(i)

for idx in idx:
    e = q.index(idx)

    if e >= (len(q)-e):
        e = len(q)-(e)
        q.rotate(e)
        # q.pop()
    else: 
        q.rotate(-e)
    q.popleft() 
    count += e   
print(count)
# 조건1. 첫 번째 원소를 뽑아낸다