# Stack
stack = []

# insert(8)-insert(3)-insert(7)-insert(5)-delete()-insert(1)-delete()
stack.append(8)
stack.append(3)
stack.append(7)
stack.append(5)
stack.pop()
stack.append(1)
stack.pop()

# 최상단 원소부터 출력
print(stack[::-1])

# 최하단 원소부터 출력
print(stack)



# Queue
from collections import deque

# 큐 구현을 위해 deque library 사용
queue = deque()

# insert(8)-insert(3)-insert(7)-insert(5)-delete()-insert(1)-delete()
queue.append(8)
queue.append(3)
queue.append(7)
queue.append(5)
queue.popleft()
queue.append(1)
queue.popleft()

# 먼저 들어온 순서대로 출력
print(queue)
# 역순으로 바꾸기
queue.reverse()
# 나중에 들어온 원소부터 출력
print(queue)



# Recursive Function
def recursive_func():
    print('재귀 함수를 호출합니다.')
    recursive_func()

recursive_func()


def recursive_function(i):
    # 5번째 호출을 했을 때 종료되도록 종료 조건 명시
    if i == 5:
        return
    print(f'{i}번째 재귀함수에서 {i+1}번째 재귀함수를 호출합니다.')

    recursive_function(i+1)
    print(f'{i}번째 재귀함수를 종료합니다.')

recursive_function(1)