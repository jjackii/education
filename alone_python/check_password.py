import re

print("The password is at least 8 characters including a number, a alphabet and a special character..")

# Sign Up
while 1:
    pw = input("Password : ")
		# pw = pw.lower()    

    if (pw.isalnum() == False) and (re.search('[a-z]+', pw)) and (re.search('[0-9]+', pw)) and (len(pw) > 7):
            print("Registered")
            break
    else:
        print("Make sure it's at least 8 characters including a number and a alphabet and a special character..")
        continue

# Sign In
print("Sign In")
for i in range(5):
    pw_input = input("Enter your password : ")

    if pw_input == pw:
        print("Signed in")
        break
    else:
        i = i + 1
        if i == 5:
            print("Please contact the administrator.")
        else:
            print(f"Invalid password.({i} times)")
            continue