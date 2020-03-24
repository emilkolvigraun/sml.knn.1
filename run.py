from _ml import KNN
from utility import Processor

def print_performance(): 
    print('\n RESULTS:')
    print(training)
    print(test)
    print(cross)

# defining filename
fnm = 'data/id_10p.csv'

# declaring processor
pro = Processor('90_10_results_10p_full')

# starting engine
obj = KNN(log=True)
knn = obj.knn
cvl = obj.cross_validation

# from k=1 to k=99 execute knn and cross validation
for i in range(100):
    pro.add(knn(filename=fnm, test_size=0.2, k=i, drop=[]), i)
    # pro.add(knn(filename=fnm, test_size=0.5, k=i, drop=[]), i)
    if i > 0 and i < 11:
        pro.add(cvl(filename=fnm, cv=10, k=5, drop=[]), 5)
# print(cvr)

# pretty prints content
# pro.pprint()

# calculate performance metrics
training, test, cross = pro.calculate_performance()
print_performance()

# generate a plot in PDF format to obtain vector graphics
pro.generate_plot()

# export the data to a CSV file
pro.export_to_csv()