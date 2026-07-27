import numpy as np
import statistics
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/Admin/Desktop/(Copy) diabetes.csv")

X = df.drop(columns=["Outcome"]).values   # convert to numpy arrays
Y = df['Outcome'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, stratify= Y, random_state=42)

print(X.shape, x_train.shape, x_test.shape)


class knn_classifier():

    def __init__(self):
        pass

    def get_distance(self, training_data_point, testing_data_point):
            
        dist = np.sum((training_data_point - testing_data_point) ** 2)
        
        return np.sqrt(dist)

    def get_nearest_neighbor(self, X_train, y_train, test_data, K):
        distance_list = []
        for i in range(len(X_train)):
            distance = self.get_distance(X_train[i], test_data)
            distance_list.append((y_train[i], distance))   # pair label with distance

        distance_list.sort(key=lambda x: x[1])

        neighbors = [distance_list[i][0] for i in range(K)]  # just the labels
        return neighbors

    def prediction(self, X_train, y_train, test_data, K):
        neighbors = self.get_nearest_neighbor(X_train, y_train, test_data, K)
        predicted_class = statistics.mode(neighbors)
        return predicted_class


if __name__ == "__main__":
    classifier = knn_classifier()

    # predict for a single test point
    single_pred = classifier.prediction(x_train, y_train, x_test[0], 7)
    print("Prediction for first test point:", single_pred, "| Actual:", y_test[0])

    # predict for the full test set
    predictions = [classifier.prediction(x_train, y_train, point, 7) for point in x_test]
    accuracy = np.mean(np.array(predictions) == y_test)
    print("Accuracy:", accuracy)