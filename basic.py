# file = open("data.txt", "w")
# file.write("hello world\n")
# file.write("file handling in python\n")
# file.close()

# file = open("data.txt", "r")
# content = file.read()
# print(content)
# file.close()

file = open("data.txt", "a")
file.write("appending new line\n")
file.close()