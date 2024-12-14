import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


CM_BRIGHT = ListedColormap(['#FF0000', '#0000FF'])


def plot_decision_boundary(X: np.array, y: np.array, model):
    x_min = X[:, 0].min() - .5
    x_max = X[:, 0].max() + .5
    y_min = X[:, 1].min() - .5
    y_max = X[:, 1].max() + .5
    h = .05  # step size in the mesh

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Zd = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Zd = Zd.reshape(xx.shape)

    Zp = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Zp = Zp.reshape(xx.shape)

    # Error de clasificación
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    plt.figure(1, figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=CM_BRIGHT)
    plt.axis([x_min, x_max, y_min, y_max])
    plt.contour(xx, yy, Zd, levels=[0], linewidths=2)
    plt.contourf(xx, yy, Zd, cmap=plt.cm.RdBu, alpha=.5)
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$x_2$", fontsize=16)
    plt.title(f'FRONTERA DECISION\n Acc: {acc:0.2g}')

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=CM_BRIGHT)
    plt.axis([x_min, x_max, y_min, y_max])
    # plt.contour(xx, yy, Zp, levels=[0], linewidths=2)
    plt.contourf(xx, yy, Zp, cmap=plt.cm.RdBu, alpha=.5)
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$x_2$", fontsize=16)
    plt.title(f'PROBABILIDAD\n Acc: {acc:0.2g}')

    plt.tight_layout()
    plt.show()


def plot_decision_boundary_poly(X, y, model, poly):
    x_min = X[:, 0].min() - .5
    x_max = X[:, 0].max() + .5
    y_min = X[:, 1].min() - .5
    y_max = X[:, 1].max() + .5
    h = .05  # step size in the mesh

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    XX = np.c_[xx.ravel(), yy.ravel()]
    Zd = model.predict(poly.fit_transform(XX))
    Zd = Zd.reshape(xx.shape)

    Zp = model.predict_proba(poly.fit_transform(XX))[:, 1]
    Zp = Zp.reshape(xx.shape)

    # Error de clasificación
    ypred = model.predict(poly.fit_transform(X))
    acc = accuracy_score(y, ypred)

    plt.figure(1, figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=CM_BRIGHT)
    plt.axis([x_min, x_max, y_min, y_max])
    plt.contour(xx, yy, Zd, levels=[0], linewidths=2)
    plt.contourf(xx, yy, Zd, cmap=plt.cm.RdBu, alpha=.5)
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$x_2$", fontsize=16)
    plt.title(f'FRONTERA DECISION\n Acc: {acc:0.2g}')

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=CM_BRIGHT)
    plt.axis([x_min, x_max, y_min, y_max])
    plt.contour(xx, yy, Zp, levels=[0], linewidths=2)
    plt.contourf(xx, yy, Zp, cmap=plt.cm.RdBu, alpha=.5)
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$x_2$", fontsize=16)
    plt.title(f'PROBABILIDAD\n Acc: {acc:0.2g}')

    plt.tight_layout()
    plt.show()


def plot_decision_boundary_svm(X, y, model):
    x_min = X[:, 0].min() - .5
    x_max = X[:, 0].max() + .5
    y_min = X[:, 1].min() - .5
    y_max = X[:, 1].max() + .5
    h = .05  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Zd = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Zd = Zd.reshape(xx.shape)

    Zp = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Zp = Zp.reshape(xx.shape)

    # Error de clasificación
    ypred = model.predict(X)
    acc = accuracy_score(y, ypred)

    plt.figure(1, figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, c='k', facecolors='none')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=CM_BRIGHT)
    plt.axis([x_min, x_max, y_min, y_max])
    plt.contour(xx, yy, Zd, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
    plt.contourf(xx, yy, Zd > 0, cmap=plt.cm.RdBu, alpha=.5)
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$x_2$", fontsize=16)
    plt.title(f'FRONTERA DECISION\n Acc: {acc:0.2g}')

    plt.subplot(1, 2, 2)
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, c='k', facecolors='none')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=CM_BRIGHT)
    plt.axis([x_min, x_max, y_min, y_max])
    plt.contour(xx, yy, Zd, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
    plt.contourf(xx, yy, Zp, cmap=plt.cm.RdBu, alpha=.5)
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$x_2$", fontsize=16)
    plt.title(f'PROBABILIDAD\n Acc: {acc:0.2g}')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(confmat):
    _, ax = plt.subplots(figsize=(7, 7))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    plt.tight_layout()
    plt.show()


def poly_linear_regression(x_i, y_i, x, y, degree):
    poly = PolynomialFeatures(degree)
    X_i = poly.fit_transform(x_i.reshape(-1, 1))
    X_test = poly.fit_transform(x.reshape(-1, 1))
    lr = LinearRegression().fit(X_i, y_i)

    y_hat = lr.predict(X_i)
    fw = lr.predict(X_test)

    error_train = np.mean(np.power(y_i-y_hat, 2))
    error_test = np.mean(np.power(y-fw, 2))

    return fw, error_test, error_train
