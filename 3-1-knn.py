#sklean kutuphane hazir veri setleri ile 
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt

# (1) veri seti incelemesi
cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data ,columns=cancer.feature_names)
df ["target"]= cancer.target



# (2) Ml model secimi _knn siniflandiricisi
# (3) train the model

X = cancer.data #feature
y = cancer.target #target

#train test split 
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3 , random_state= 42)


#olceklendirme 
scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)

#knn modelini train et
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train) #fit bizim verdigimiz etiketlerle modeli egitiyo


# (4) sonuclarin degerlendirmesi
y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("dogruluk",accuracy)

conf_matrix = confusion_matrix(y_test,y_pred)
print("confusion matrix:")
print(conf_matrix)


# (5) hiperparametre ayarlamasi
accuracy_values =[]
k_value = []
for k in range(1,21):
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(X_train ,y_train)
   y_pred = knn.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   accuracy_values.append(accuracy)
   k_value.append(k)

plt.figure()
plt.plot(k_value,accuracy_values,marker ="o",linestyle ="-")
plt.title("k degerine gore dogruluk")
plt.xlabel("k degeri")
plt.ylabel("dogruluk")
plt.xticks(k_value)
plt.grid(True)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

X = np.sort(5 * np.random.rand(40, 1),axis= 0) #features
y = np.sin(X).ravel() #target

#plt.scatter(X, y)

#add noise 
y[::5] += 1 * (0.5 - np.random.rand(8))
#plt.scatter(X, y)
T = np.linspace(0,5,500)[:,np.newaxis]

for weights in ["uniform","distance"]:
   
   knn =KNeighborsRegressor(n_neighbors=5,weights=weights)
   y_pred = knn.fit(X, y).predict(T)



   plt.Figure()
   plt.scatter(X, y, color ="green",label = "data")
   plt.plot(T,y_pred,color = "blue",label = "prediction")
   plt.axis("tight")
   plt.legend()
   plt.title("knn regressor weights = {}".format(weights))

plt.tight_layout()
plt.show()































