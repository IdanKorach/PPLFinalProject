import oti

while True:
    text = input('oti > ')
    result, error = oti.run('<stdin>', text) # stdin for the fn as placeholder 

    if error: 
        print(error.as_string())
    else: 
        print(result)