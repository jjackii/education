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



# 팩토리얼 구현 예제
# n! = 1*2*3*...*(n-1)*n, 0! and 1! == 1

# Iteration
def factorial_iterative(n):
    result = 1

    # 1부터 n까지 차례대로 곱하기
    for i in range(1, n+1):
        result *= i
    return result

# Recursive
def factorial_recursive(n):
    # n이 1 이하인 경우 1을 반환
    if n <= 1:
        return 1
    
    # n! = n*(n-1)!
    return n * factorial_recursive(n-1)

print(f'반복적으로 구현: {factorial_iterative(5)}')
print('재귀적으로 구현:', factorial_recursive(5))



# 최대공약수 계산 (유클리드 호제법) 예제
def gdc(a, b):
    if a%b == 0:
        return b
    else:
        return gdc(b, a%b)

print(gdc(192, 162))