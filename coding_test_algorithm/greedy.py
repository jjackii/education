## 이것이 코딩 테스트다 - Greedy

# ------ sol(1). 거스름 돈
n = 1260
count = 0

coin = [500, 100, 50, 10] # 배수

for c in coin:
    count += n//c # 2(500)
    n %= c # 1260%(500)=260 

print(count)



## ------ sol(2). 1이 될 때까지
n, k = map(int, input().split())
count=0

while n != 1:
    if n%k==0:
        n = n//k
        count+=1
    else:
        n-=1
        count+=1
    # print(n)
print(count)

## ------
n, k = map(int, input().split())
result=0

while True:
    # 가장 가까운 나누어 떨어지는 수
    target = (n//k) * k

    # 총 연산 횟수 (1을 빼는 계산을 몇 번 할지)
    result += (n - target)

    n = target

    if n < k:
        break
    
    # 더 나눌 수 o
    # result(1을 빼는 계산 횟수) + 1
    result += 1 

    # 나누어 떨어짐(위에서 1을 빼는 계산 count)
    n //= k 

# 마지막으로 남은 수에 대해 1씩 빼기 / ex) n=3, 1이 될 때까지 빼면 총 2(3-1)번 뺌
result += (n-1) 

print(result)



## ------ sol(3). 곱하기 혹은 더하기
n = list(map(int, input().split()))
# print(type(n))

result = 0

for i in range(len(n)):
    if i==0:
        result+=(n[0])
    elif n[i] > 1: 
        result*=n[i]
    else:
        result+=n[i]
        
print(result) 
# list에 '0'이 들어가면 X



## ------
data = input()

result = int(data[0])

for i in range(1, len(data)):
    num = int(data[i])

    if num <= 1 or result <= 1:
        result += num
    else:
        result *= num

print(result)


