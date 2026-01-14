

# %%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# %%
n = 240
gap = 2.2
std = 0.7

n1 = n // 2
n2 = n - n1

X1 = np.random.normal(loc=(-gap, -gap), scale=std, size=(n1, 2))
X2 = np.random.normal(loc=( gap,  gap), scale=std, size=(n2, 2))

X = np.vstack([X1, X2])
y = np.hstack([np.full(n1, -1, dtype=int), np.full(n2, +1, dtype=int)])

print("X shape:", X.shape)
print("y counts:", {c:int((y==c).sum()) for c in np.unique(y)}) # 120 data is labeled as -1, 120 of them is +1

# %% 
plt.figure(figsize=(5,5))
plt.scatter(X[y==-1,0], X[y==-1,1], s=25, label="-1")
plt.scatter(X[y==+1,0], X[y==+1,1], s=25, label="+1")
plt.title("Raw 2D data")
plt.legend()
plt.show()

# %%
mean = X.mean(axis=0)
std_deviation = X.std(axis=0)
std_deviation[std_deviation == 0] = 1.0

Xs = (X - mean) / std_deviation

print("mean after standardize (approx):", Xs.mean(axis=0))
print("std  after standardize (approx):", Xs.std(axis=0))

# %%

idx = np.arange(Xs.shape[0])
np.random.shuffle(idx)

# 4:1 split 

test_size = 0.25
cut = int(len(idx) * (1 - test_size))
train_idx = idx[:cut]
test_idx  = idx[cut:]

Xtr = Xs[train_idx]
ytr = y[train_idx]
Xte = Xs[test_idx]
yte = y[test_idx]

print("train:", Xtr.shape, "test:", Xte.shape)
# %%

d = Xtr.shape[1]
print(d)
w = np.zeros(d, dtype=float)
print(w)

b = 0.0
print(b)

lr = 1.0
epochs = 50

print("w:", w, "b:", b)
print("lr:", lr, "epochs:", epochs)

# %%
mistakes_per_epoch = []

for ep in range(epochs):
    order = np.arange(Xtr.shape[0]) # for every epoch training X 
    np.random.shuffle(order) # shuffle training set  

    mistakes = 0

    for i in order: # for every sample in training set 
        x_i = Xtr[i] 
        y_i = ytr[i]

        s = float(np.dot(w, x_i) + b)  # s = w·x + b

        if y_i * s <= 0:  # and here it is check if result is correct or not. HOW ? if s(prediction) is equal to y(eqtual) then it returns >= 0 every time. (both -, - or +, +)
            w = w + lr * y_i * x_i  # and makes small changes on weight and bias
            b = b + lr * y_i
            mistakes += 1

    mistakes_per_epoch.append(mistakes) # and keep track of mistakes 

    if mistakes == 0:
        break

print("epochs run:", len(mistakes_per_epoch))
print("mistakes per epoch:", mistakes_per_epoch)
print("final w:", w, "final b:", b)

# %%

s_tr = Xtr @ w + b
yhat_tr = np.where(s_tr >= 0, 1, -1)
train_acc = (yhat_tr == ytr).mean()

s_te = Xte @ w + b
yhat_te = np.where(s_te >= 0, 1, -1)
test_acc = (yhat_te == yte).mean()

print("train accuracy:", train_acc)
print("test  accuracy:", test_acc)


plt.figure(figsize=(6,4))
plt.plot(mistakes_per_epoch, marker="o")
plt.title("Perceptron mistakes per epoch")
plt.xlabel("Epoch")
plt.ylabel("Mistakes")
plt.show()

# %%
x_min, x_max = Xs[:,0].min() - 0.8, Xs[:,0].max() + 0.8
y_min, y_max = Xs[:,1].min() - 0.8, Xs[:,1].max() + 0.8

step = 0.03
xs = np.arange(x_min, x_max, step)
ys = np.arange(y_min, y_max, step)
xx, yy = np.meshgrid(xs, ys)
grid = np.c_[xx.ravel(), yy.ravel()]

s_grid = grid @ w + b
pred_grid = np.where(s_grid >= 0, 1, -1).reshape(xx.shape)

plt.figure(figsize=(6,5))
plt.contourf(xx, yy, pred_grid, alpha=0.25)

plt.scatter(Xte[yte==-1,0], Xte[yte==-1,1], s=25, label="-1")
plt.scatter(Xte[yte==+1,0], Xte[yte==+1,1], s=25, label="+1")

plt.title("Perceptron decision regions (test points)")
plt.legend()
plt.show()

# %%
