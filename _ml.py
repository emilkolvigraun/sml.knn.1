import pandas, math, numpy, time
from sklearn import metrics

class KNNResponse:

    def __init__(self, train_result, test_result, test_time, training_time, fit_time):
        self.train_result = round(train_result, 4)
        self.test_result = round(test_result, 4)
        self.test_time = test_time
        self.training_time = training_time
        self.fit_time = fit_time

    def __repr__(self):
        return 'KNN(train_result=%s percent, test_result=%s percent, training_time=%s ms, test_time=%s ms, fit_time=%s ms)'%(str(self.train_result), str(self.test_result), str(self.training_time), str(self.test_time), str(self.fit_time))

class CrossResponse:

    def __init__(self, mean, std, time):
        self.mean = round(mean, 4)
        self.std = std
        self.time = time

    def __repr__(self):
        return 'Cross(mean=%s, std=%s, time=%s ms)'%(str(self.mean), str(self.std), str(self.time))

class KNN:

    def __init__(self, log:bool=True):
        self.log_level = log

    def log(self, tag, msg, nl=False):
        if nl:
            message = '\n%s --- %s : %s'%(str(round(time.time())), str(tag), str(msg))
        else:
            message = '%s --- %s : %s'%(str(round(time.time())), str(tag), str(msg))
        if self.log_level: print(message)

    def accuracy_metric(self, results, truth_count):
        return metrics.accuracy_score(results, truth_count)

    def get_time(self, ):
        return int(round(time.time() * 1000))

    def test(self, x, y, train, test, k):
        
        from sklearn.neighbors import KNeighborsClassifier
        # defining the classifier with k
        knn = KNeighborsClassifier(n_neighbors=k)

        t0 = self.get_time()
        # fitting the model to the training data
        knn.fit(train[x], numpy.ravel(train[y]))

        fit_time = self.get_time() - t0
        self.log('testing fit time', fit_time)

        t0 = self.get_time()
        test_predictions = knn.predict(test[x])
        test_time = self.get_time() - t0
        self.log('testing time', test_time)

        test_accuracy = self.accuracy_metric(test_predictions, test[y])
        self.log('testing accuracy','%s percent'%round(test_accuracy*100,2))

        return test_accuracy, test_time, fit_time

    def train(self, x, y, train, k):
        
        from sklearn.neighbors import KNeighborsClassifier
        # defining the classifier with k
        knn = KNeighborsClassifier(n_neighbors=k)

        t0 = self.get_time()
        # fitting the model to the training data
        knn.fit(train[x], numpy.ravel(train[y]))

        fit_time = self.get_time() - t0
        self.log('training fit time', fit_time)

        t0 = self.get_time()
        train_predictions = knn.predict(train[x])
        train_time = self.get_time() - t0
        self.log('training time', train_time)

        train_accuracy = self.accuracy_metric(train_predictions, train[y])
        self.log('training accuracy','%s percent'%round(train_accuracy*100,2))

        return train_accuracy, train_time, fit_time

    def algorithm(self, x, y, train, test, k):
        train_accuracy, train_time, fit_time_train = self.train(x, y, train, k)
        test_accuracy, test_time, fit_time_test = self.test(x, y, train, test, k)
        return KNNResponse(train_accuracy, test_accuracy, train_time, test_time, (fit_time_test+fit_time_train)/2)

    def load_data(self, filename:str, drop:list):
        return pandas.read_csv(filename).drop(columns=drop)

    def test_slice_shuffle(self, data, size:float):
        return numpy.random.permutation(data.index), math.floor(len(data)*size)

    def get_evaluation_metric(self, data):
        eval = [data.columns.values[0]]
        self.log('eval metric', eval)
        return [data.columns.values[0]]

    def get_predict_data(self, data):
        return data.columns.values[1:]

    def knn(self, filename:str, test_size:float=0.2, k:int=5, drop:list=[]):
        if k < 1:
            self.log('skipping because neighbours is <=0', filename, True)
            return
        else:
            self.log('starting knn on', filename, True)

        # load the dataset
        data = self.load_data(filename, drop)

        # randomize indexes and calculate the test sixes
        shuffle, test_size = self.test_slice_shuffle(data, test_size)

        # declare test and training set
        test = data.loc[shuffle[1:test_size]]
        train = data.loc[shuffle[test_size:]]

        # the columns we are making the prediction with
        x = self.get_predict_data(data)

        # the column(s) we are trying to predict
        y = self.get_evaluation_metric(data)

        return self.algorithm(x, y, train, test, k)

    def cross_validation(self, filename:str, cv:int=10, k:int=5, drop:list=[]):
        from sklearn.model_selection import cross_val_score
        if k < 1:
            self.log('skipping because neighbours is <=0', filename, True)
            return
        else:
            self.log('starting cross validation on', filename, True)

        data = self.load_data(filename, drop)
        
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=k)

        # the columns we are making the prediction with
        x = self.get_predict_data(data)

        # the column(s) we are trying to predict
        y = self.get_evaluation_metric(data)

        t0 = self.get_time()
        scores = cross_val_score(knn, data[x], numpy.ravel(data[y]), cv=cv)
        cross_time = self.get_time() - t0

        mean = numpy.mean(scores)
        self.log('mean', mean)
        std = numpy.std(scores) 
        self.log('standard deviation', std)
        ctime= cross_time
        self.log('computation time', ctime)

        return CrossResponse(mean, std, ctime)

        