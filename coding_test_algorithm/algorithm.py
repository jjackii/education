# ------ Q(1).
n = 1260
count = 0

coin = [500, 100, 50, 10] # 배수

for c in coin:
    count += n//c # 2(500)
    n %= c # 1260%(500)=260 

print(count)



## ------ Q(2).
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

    result += 1 
    n //= k 

# 마지막으로 남은 수에 대해 1씩 빼기
result += (n-1) 

print(result)





