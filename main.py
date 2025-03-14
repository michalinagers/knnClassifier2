import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

iris = pd.read_csv('/content/drive/MyDrive/week_6/iris.csv')

fig, ax = plt.subplots(figsize=(10,10))
seaborn.histplot(iris['petal_width'], ax=ax)
iris['species'].value_counts().plot(kind="bar")
seaborn.pairplot(iris)

x = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

knn_model = KNeighborsClassifier(n_neighbors=6)
knn_model.fit(x_train, y_train)
knn_model.score(x_test, y_test)
knn_model.predict(x_test[10:])
knn_model.predict([[110, 5.6, 7.9, 0.1]])

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
knn_scaled = KNeighborsClassifier()
knn_scaled.fit(x_train_scaled, y_train)
knn_scaled.score(x_test_scaled, y_test)

x = iris[['petal_length', 'petal_width', 'sepal_length', 'petal_width']]
y = iris['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

def plot_fruit_knn(x, y, n_neighbors, weights):
    x_mat = x[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y_mat = y.values
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#AFAFAF'])
    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(x_mat, y_mat)
    mesh_step_size = .01
    plot_symbol_size = 50
    x_min, x_max = x_mat[:, 0].min() - 1, x_mat[:, 0].max() + 1
    y_min, y_max = x_mat[:, 1].min() - 1, x_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(x_mat[:, 0], x_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor='black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    patch0 = mpatches.Patch(color='#FF0000', label='apple')
    patch1 = mpatches.Patch(color='#00FF00', label='mandarin')
    patch2 = mpatches.Patch(color='#0000FF', label='orange')
    patch3 = mpatches.Patch(color='#AFAFAF', label='lemon')
    plt.legend(handles=[patch0, patch1, patch2, patch3])
    plt.xlabel('height (cm)')
    plt.ylabel('width (cm)')
    plt.title("4-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
    plt.show()

plot_fruit_knn(x_train, y_train, 5, 'uniform')
print(iris)
