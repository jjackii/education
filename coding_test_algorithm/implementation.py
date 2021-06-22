## Implementation

## ------ sol(1). 상하좌우 (Simulation)

# N x N                 # X : n = map(int, input().split()) 
n = int(input())
plans = input().split()

x, y = 1, 1

# R, L, U, D
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

move_types = ['R', 'L', 'U', 'D']

# 이동 계획 확인, 하나씩
for plan in plans:

    # 이동 후 좌표 구하기
    for i in range(len(move_types)):
        if plan == move_types[i]:
            nx = x + dx[i]
            ny = y + dy[i]

    # 공간을 벗어나는 경우 무시
    if nx < 1 or ny < 1 or nx > n or ny > n:
        continue

    # 이동
    x, y = nx, ny

print(x, y)