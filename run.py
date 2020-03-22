from _ml import KNN
from utility import Processor

def print_performance(): 
    print('\n RESULTS:')
    print(training)
    print(test)
    print(cross)

# defining filename
fnm = 'data/id_1p.csv'

# declaring processor
pro = Processor('results/plot')

# starting engine
obj = KNN(log=True)
knn = obj.knn
cvl = obj.cross_validation

# from k=1 to k=99 execute knn and cross validation
for i in range(100):
    pro.add(knn(filename=fnm, test_size=0.1, k=i, drop=['index']), i)
    # pro.add(cvl(filename=fnm, cv=10, k=i, drop=['index']), i)

# pretty prints content
pro.print()

# calculate performance metrics
training, test, cross = pro.calculate_performance()
print_performance()

# generate a plot in PDF format to obtain vector graphics
pro.generate_plot()

# export the data to a CSV file
pro.export_to_csv()