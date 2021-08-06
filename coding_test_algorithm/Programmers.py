# Challenge 1
def solution(price, money, count):
    # answer = -1
    answer=0
    result=0 

    for i in range(count):
        result += price*(i+1)

    answer = result - money

    if answer < 0:
        answer = 0 
				# answer == 0 :error

    return answer