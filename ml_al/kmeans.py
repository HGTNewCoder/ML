'''
Đầu vào:
Dữ liệu X và số lượng cluster cần tìm K

Đầu ra:
Các center M và label vector cho từng điểm dữ liệu Y

Chọn 
1 K điểm bất kỳ làm các center ban đầu.
2 Phân mỗi điểm dữ liệu vào cluster có center gần nó nhất.
3 Nếu việc gán dữ liệu vào từng cluster ở bước 2 không thay đổi so với vòng lặp trước nó thì ta dừng thuật toán.
4 Cập nhật center cho từng cluster bằng cách lấy trung bình cộng của tất các các điểm dữ liệu đã được gán vào cluster đó sau bước 2.
Quay lại bước 2.
'''
'''
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100),
    np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 100),
    np.random.multivariate_normal([10, 0], [[1, 0], [0, 1]], 100)
])

#Init PreData
n = data.shape[0]
k = 3
M = np.random.random((k, 2))
Y = np.zeros((n, k, 1))
flag = False

def norm(A: np.ndarray, B: np.ndarray):
    return np.sqrt(pow((A[0] - B[0]), 2) + pow((A[1] - B[1]), 2))

while flag == False:
    Copy = M
    for i in range(n):
        min = norm(data[i], M[0])
        id = 0
        for j in range (k):
            if norm(data[i], M[j]) <= min:
                min = norm(data[i], M[j])
                id = j
        Y[i][id][0] = 1
    for j in range (k):
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for i in range (n):
            sum1 += Y[i][j][0] * data[i][0]
            sum2 += Y[i][j][0] * data[i][1]
            sum3 += Y[i][j][0]
        M[j][0] = sum1 / sum3
        M[j][1] = sum2 / sum3
    for j in range (k):
        if Copy[j][0] == M[j][0] and Copy[j][1] == M[j][1]:
            flag = True
print(M)
        
plt.scatter(data[:, 0], data[:, 1], s=50)
plt.scatter(M[:, 0], M[:, 1], s=100)
plt.xlabel("F1")
plt.ylabel("F2")
plt.title("Dataset")
plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100),
    np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 100),
    np.random.multivariate_normal([10, 0], [[1, 0], [0, 1]], 100)
])

# Initialize data
n = data.shape[0]
k = 3
M = np.random.rand(k, 2)
Y = np.zeros(n, dtype=int)
tol = 1e-4
max_iters = 1000

def norm(A: np.ndarray, B: np.ndarray):
    return np.linalg.norm(A - B)

for iteration in range(max_iters):
    # Step 2: Assign each point to the nearest center
    for i in range(n):
        distances = np.array([norm(data[i]   , M[j]) for j in range(k)])
        Y[i] = np.argmin(distances)

    # Step 4: Update centers
    new_M = np.zeros_like(M)
    for j in range(k):
        points_in_cluster = data[Y == j]
        if len(points_in_cluster) > 0:
            new_M[j] = points_in_cluster.mean(axis=0)
        else:
            new_M[j] = M[j]  # Keep the same center if no points are assigned to the cluster

    # Check for convergence
    if np.all(np.abs(new_M - M) < tol):
        break
    M = new_M

print(M)

plt.scatter(data[:, 0], data[:, 1], s=50)
plt.scatter(M[:, 0], M[:, 1], s=100, c='red')
plt.xlabel("F1")
plt.ylabel("F2")
plt.title("Dataset")
plt.show()
