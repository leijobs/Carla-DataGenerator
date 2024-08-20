import matplotlib.pyplot as plt

def showPc(x, y):
    plt.figure(figsize=(10,10))
    plt.scatter(x, y, s=5, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)