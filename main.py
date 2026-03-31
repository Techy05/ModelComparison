from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
Y = iris.target

# print(iris.feature_names)
# print(iris.target_names)
print("Complete Dataset:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=95
)

print("Training data:", X_train.shape) 
print("Testing data:", X_test.shape)
# print(pd.DataFrame(X_scaled, columns=iris.feature_names).describe())

################# LOGISTIC REGRESSION ######################

lr_model = LogisticRegression(random_state=95)
lr_model.fit(X_train, Y_train)

lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(Y_test, lr_predictions)

print(f"Logistic Regression Accuracy: {lr_accuracy*100}%")


########################### KNN ###########################

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)

knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(Y_test, knn_predictions)

print(f"KNN Accuracy: {knn_accuracy*100}%")


###################### Decision Tree ######################

dt_model = DecisionTreeClassifier(random_state=95)
dt_model.fit(X_train, Y_train)

dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(Y_test, dt_predictions)

print(f"Decision Tree Accuracy: {dt_accuracy*100}%")
