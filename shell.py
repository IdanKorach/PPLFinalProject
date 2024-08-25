import oti
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python shell.py <filename>")
        return

    filename = sys.argv[1]
    
    try:
        with open(filename, 'r') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return
    
    result, error = oti.run(filename, text)
    if error:
        print(error.as_string())

if __name__ == "__main__":
    main()
