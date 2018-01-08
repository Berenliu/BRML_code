import numpy as np

r = 0.005

def gradient(v, x, x0):
    n = len(v)
    d = np.sum((x-x0)**2, axis=1)
    d = d+0.1
    a = np.multiply(4/d**2, v-1/d).reshape(1, -1)
    return 1./n*(np.dot(a, x-x0))

def load_data():
    point_list = []
    value_list = []
    f = open("explosition_data.csv", "rb")
    for line in f:
        fields = [float(i) for i in line.strip().split()]
        point_list.append((fields[0], fields[1]))
        value_list.append(fields[2])
    f.close()
    return np.array(point_list), np.array(value_list)


def find_explosition_point(x, v):
    x0 = np.array([0.0,-0.1])
    x1 = np.array([1.0,0.0])
    k = 1
    while (np.sum((x0-x1)**2)>1e-6):
        print("%d epoch" % k)
        print(x0)
        x1 = x0
        grad = gradient(v, x, x1)
        print(grad)
        x0 = x0 + r*grad
        k += 1
    print(x0)

if __name__ == "__main__":
    x, v = load_data()
    print(x.shape)
    print(v.shape)
    find_explosition_point(x, v)

        
        

