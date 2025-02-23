# Bai 1

from traceback import print_tb

from numpy.ma.core import identity

print("Bai 1", end="\n")
name = input("Nhap ten cua ban: ")

for char in name:
    print(char, end=" ")

print(end="\n")
#Bai 2
print("bai 2", end="\n")
print("Tat ca so le tu 1 den 10")
for i in range(1, 11):
    if i % 2 != 0:
        print(i, end=" ")
print(end="\n")

#Bai 3
print("Bai 3", end="\n")
tong = 0
for i in range(1, 11):
    if i % 2 != 0:
        tong += i
print(f"Tong le: {tong}")

print("Tong so tu 1 - 6:", end="\n")

tongAll = 0
for i in range(1, 7):
    tongAll += i
print(f"Tong = {tongAll}", end="\n")

#Bai 4:
print("Bai 4:", end="\n")
mydict = {"a":1, "b":2, "c":3, "d":4}
for key in mydict.keys():
    print(key, end=" ")
print(end="\n")
for values in mydict.values():
    print(values, end=" ")
print(end="\n")
for key, values in mydict.items():
    print(f"Key = {key}, Values = {values}", end="\n")

#Bai 5
#5/ Given courses=[131,141,142,212]
# and names=[“Maths”,”Physics”,”Chem”, “Bio”].
# Print a sequence of tuples, each of them contains one courses
# and one names

print("bai 5", end="\n")
courses=[131,141,142,212]
names=["Maths", "Physic", "Chem", "Bio"]

zipped = zip(courses, names)
print(list(zipped), end="\n")

#Bai 6
#Find the number of consonants in “jabbawocky” by two ways
	#a/ Directly (i.e without using the command “continue”)
	#b/ Check whether it’s characters are in vowels set and using the command “continue”
print("Bai 6", end="\n")
print("without continue", end="\n")
vowels = "aeiou"
for char in "jabbawocky":
    if char not in vowels:
        print(char, end= " ")
print(end="\n")
#using continue
print("using continue", end="\n")
totals = 0
for char in "jabbawocky":
    if char in vowels:
        continue
    print(f"{char}", end=" ")
print(end="\n")
#Bai 7
#7/ a is a number such that -2<=a<3. Print out all the results of 10/a using try…except.
# When a=0, print out “can’t divided by zero”

print("Bai 7", end="\n")
for a in range(-2, 3):
    try:
        print(10/a)
    except ZeroDivisionError:
        print("Can’t divided by zero")

#Bai 8
#8/ Given ages=[23,10,80]
#And names=[Hoa,Lam,Nam]. Using lambda function to sort a list containing tuples
# by increasing of the ages
print("Bai 8", end="\n")
ages = [23, 10, 80]
names = ["Hoa", "Lam", "Nam"]
data = zip(ages, names)
print(sorted(data, key=lambda x : x[0]), end="\n")

#Bai 9
print("Bai 9", end="\n")
# imput_file = open("firstname.txt")
# for line in imput_file:
#     print(line, end="")
input_file = open("firstname.txt")
first_names = input_file.read()
print(first_names)
input_file.close()

#define a function
print("\n")
print("DEFINE A FUNCTION", end="\n")

def sumNumber(x, y):
    return x + y

print(f"Tong = {sumNumber(3, 4)}")

print("\nBai 2", end="\n")
import numpy as np

def matrix_operations(M, v):
    rank_M = np.linalg.matrix_rank(M)
    shape_M = M.shape
    shape_v = v.shape

    print("Matrix M:\n", M)
    print("\nVector v:\n", v)
    print("\nRank of matrix M:", rank_M)
    print("\nShape of matrix M:", shape_M)
    print("\nShape of vector v:", shape_v)

    if rank_M == shape_M[0]:
        print("\nMa tran fullrank")
    else:
        print("\nMa tran is not full rank.")

    try:
        x = np.linalg.solve(M, v)
        print("\nSolution to Mx = v:\n", x)
        print("\nVector v is in the column space of M.")
    except np.linalg.LinAlgError:
        print("\nNo solution to Mx = v. Vector v is not in the column space of M.")

    print("\nBai 3\n")

    M_new = M + 3

    print("Ma trận M ban đầu:\n", M)
    print("\nMa trận M mới:\n", M_new)

    print("\nBai 4\n")

    M_T = M.T

    # Chuyển vị của vector v (iến thành ma trận cột
    v_T = v.reshape(3, 1)

    print("Ma trận M:\n", M)
    print("\nChuyển vị của M (M_T):\n", M_T)
    print("\nVector v:\n", v)
    print("\nChuyển vị của v (v_T):\n", v_T)

M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

v = np.array([1, 2, 3])

matrix_operations(M, v)

print("\nBai 5\n")
# Vector x
x = np.array([2, 7])

# Tính định mức của x
norm_x = np.linalg.norm(x)

# Chuẩn hóa vector x
x_normalized = x / norm_x

print("Vector x:", x)
print("Định mức của x:", norm_x)
print("Vector x đã chuẩn hóa:", x_normalized)

print("\nBai 6\n")
a = np.array([10, 15])
b = np.array([8, 2])
c = np.array([1, 2, 3])

print("a + b =", a + b)
print("a - b =", a - b)

try:
  print("a - c =", a - c)
except ValueError as e:
  print("Lỗi:", e)

print("\nBai 7\n")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("Ma tran a: ", a, end="\n")
print("ma tran b: ", b, end="\n")
print("Tich vo huong cua hai ma tran: ", np.dot(a, b))

print("\nBai 8\n")

A = np.array([[2, 4, 9],
                [3, 6, 7]])

print("\n Ma Tran A: \n", A, end="\n")
rank_A = np.linalg.matrix_rank(A)
shape_A = A.shape

print("Hạng của A:", rank_A)
print("Hình dạng của A:", shape_A)

gia_tri_7 = A[1, 2]

print("Giá trị 7 của A:", gia_tri_7, end="\n")
cot_hai_of_A = A[:, 1]
print("Cột thứ hai cua A: ", cot_hai_of_A, end="\n")


print("\nBai 9\n")

matrix = np.random.randint(-10, 10, size=(3, 3))
print("Ma tran random\n", matrix)

print("\nBai 10\n")
identify_matrix = np.eye(3)
print(identify_matrix)

print("\nBai 11\n")

matrix_11 = np.random.randint(1, 11, size=(3, 3))
print("Ma tran random\n", matrix_11)

one_command = np.trace(matrix)

print("\nVết ma trận (một lệnh):", one_command)

# b/ Tính vết bằng vòng lặp for
trace_loop = 0
for i in range(3):
    trace_loop += matrix[i, i]

print("\nVết ma trận (vòng lặp for):", trace_loop)


print("\nBai 12\n")
#duong cheo chinh
matrix_dcc = np.diag([1, 2, 3])
print(matrix_dcc)

print("\nBai 13\n")

matrix_A = np.array([[1, 1, 2],
                    [2, 4, -3],
                    [3, 6, -5]])
print("Ma tran A: \n", matrix_A)
matrix_dinhthuc = np.linalg.det(matrix_A)
print("Ma Tran dinh thuc: \n", matrix_dinhthuc)

print("\nBai 14\n")

a1 = np.array([1, -2, -5])
a2 = np.array([2, 5, 6])
M = np.column_stack((a1, a2))
print("Ma trận M:\n", M)

print("\nBai 15\n")
import matplotlib.pyplot as plt
import numpy as np

# Tạo dữ liệu
x = np.arange(-5, 6)
y = x**2


plt.plot(x, y, label="Đồ Thị")

# Đặt tiêu đề và nhãn
plt.title("Đồ thị hàm số y = x^2")
plt.xlabel("x")
plt.ylabel("y")

plt.grid(True)
plt.show()

print("\nBai 16\n")

start = 0
stop = 32
num = 4

values = np.linspace(start, stop, num)
print(values)

print("\nBai 17\n")

x = np.linspace(-5, 5, 50)
y = x**2

# Vẽ đồ thị
plt.plot(x, y)
plt.title("Đồ thị hàm số y = x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

print("\nBai 18\n")
# Tạo dữ liệu x
x = np.linspace(-5, 5, 100)  # 100 điểm dữ liệu từ -5 đến 5
y = np.exp(x)
plt.plot(x, y)
plt.title("Đồ thị hàm số y = e^x")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


print("\nBai 19\n")
x = np.linspace(0.001, 5, 100)
y = np.log(x)
plt.plot(x, y)
plt.title("Đồ thị hàm số y = log(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print("\nBai 20\n")
x = np.linspace(-5, 5, 100)
x_log = np.linspace(0.001, 5, 100)

y1 = np.exp(x)
y2 = np.exp(2*x)

y3 = np.log(x_log)
y4 = np.log(2*x_log)

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(x, y1, label="y = exp(x)")
plt.plot(x, y2, label="y = exp(2*x)")
plt.title("Đồ thị hàm số y = exp(x) và y = exp(2*x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x_log, y3, label="y = log(x)")
plt.plot(x_log, y4, label="y = log(2*x)")
plt.title("Đồ thị hàm số y = log(x) và y = log(2*x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()