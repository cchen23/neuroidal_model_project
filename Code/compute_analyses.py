import math
def T(r, p, j):
    return math.factorial(r) / (math.factorial(j) * math.factorial(r-j)) * (p ** j) * (1-p) ** (r-j)

def B(r, p, k):
    sum = 0.0
    for j in range(int(k), int(r+1)):
        sum += T(int(r), p, j)
    return sum

def compute_JOIN_probabilities(r, n, d, k):
    p = d / n
    print("2.1, should be %0.2f: %0.2f" % (r, n * B(r,p,k) ** 2))
    print("2.2, should be 0: %0.2f" % (B(n, B(r/2,p,k)*B(r,p,k),r/10)))
    pprime = (1-B(r,p,k)) * (B(r,p,k)) ** 2
    print("2.3, should be 1: %0.2f" % (B(n,pprime,2*r/3)))

def compute_LINK_probabilities(r, n, d, k):
    p = d / n
    print("2.4, should be %0.2f: %0.2f" % (k, B(n,p*B(r,p,k),k)))
    print("2.5, should be 0: %0.2f" % (B(r,B(n,p*B(r/2,p,k),k),r/2)))
    pprime = (1-B(r,p,k)) * (B(r,p,k)) ** 2
    pdoubleprime = B(n,pprime*B(r,p,k),k)
    print("2.6, should be 0: %0.2f" % (B(r,pdoubleprime,r/2)))
