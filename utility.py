import pandas

from knn import KNNResponse, CrossResponse

class Processor:

    def __init__(self, filename):
        self.filename = filename
        self.knn = {'train_result':[], 'test_result':[], 'training_time':[], 'test_time':[], 'k':[]}
        self.cross = {'mean':[], 'std':[], 'time':[], 'k':[]}

    def add(self, response, k):
        if response is not None:
            if isinstance(response, KNNResponse):
                self.knn['train_result'].append(response.train_result)
                self.knn['test_result'].append(response.test_result)
                self.knn['training_time'].append(response.training_time)
                self.knn['test_time'].append(response.test_time)
                self.knn['k'].append(k)
            elif isinstance(response, CrossResponse):
                self.cross['mean'].append(response.mean)
                self.cross['std'].append(response.std)
                self.cross['time'].append(response.time)
                self.cross['k'].append(k)
    def print(self):
        print('\nKNN data:')
        print(pandas.DataFrame.from_dict(self.knn))
        print('\nCross Validation data:')
        print(pandas.DataFrame.from_dict(self.cross))


    def export_to_csv(self):
        if len(self.knn['test_time']) > 0:
            pandas.DataFrame.from_dict(self.knn).to_csv('exported/'+self.filename+'_knn.csv')
        if len(self.cross['time']) > 0:
            pandas.DataFrame.from_dict(self.cross).to_csv('exported/'+self.filename+'_cval.csv')
        self.knn.clear()
        self.cross.clear()

    def calculate_performance(self):
        import numpy
        test_time = numpy.array(self.knn['test_time'])
        training_time = numpy.array(self.knn['training_time'])
        test_result = numpy.array(self.knn['test_result'])
        train_result = numpy.array(self.knn['train_result'])
        train_stats = 'training: result MEAN=%s, training result STD=%s, training time MEAN=%s, training time STD=%s'%(str(numpy.mean(train_result)), str(numpy.std(train_result)), str(numpy.mean(training_time)), str(numpy.std(training_time)))
        test_stats = 'test: result MEAN=%s, test result STD=%s, test time MEAN=%s, test time STD=%s'%(str(numpy.mean(test_result)), str(numpy.std(test_result)), str(numpy.mean(test_time)), str(numpy.std(test_time)))
        
        mean = numpy.array(self.cross['mean'])
        time = numpy.array(self.cross['time'])
        cross_stats = 'cross: result result MEAN=%s, result STD=%s, time MEAN=%s'%(str(numpy.mean(mean)), str(numpy.std(mean)), str(numpy.mean(time)))
        return train_stats, test_stats, cross_stats

    def generate_plot(self):
        import pandas, matplotlib.pyplot as plt, matplotlib.pylab as pylab
        from matplotlib.backends.backend_pdf import PdfPages

        params = {'axes.labelsize':'large'}
        pylab.rcParams.update(params)

        accuracy_col = "#1F618D"
        time_col = "#A04000"
        
        for extention in ['__knn', '__cval']:
            if extention == '__knn':
                df = pandas.DataFrame.from_dict(self.knn)
                res = 'test_result'
                t = 'test_time'
            elif extention == '_cval':
                df= pandas.DataFrame.from_dict(self.cross)
                res = 'mean'
                t = 'time'

            fig, ax1 = plt.subplots()

            ax1.plot(df['k'], df[res]*100, c=accuracy_col)
            ax1.tick_params(axis='y', labelcolor=accuracy_col)
            ax1.set_ylabel("[Accuracy %]", color=accuracy_col)
            ax1.xaxis.grid()

            ax2 = ax1.twinx()
            ax2.plot(df['k'], df[t], c=time_col)
            ax2.tick_params(axis='y', labelcolor=time_col)
            ax2.set_ylabel("[Computational Time ms]", color=time_col, )
            ax1.set_xlabel("K-value")

            plt.tight_layout()
            pp = PdfPages(self.filename+extention+'.pdf') 
            plt.savefig(pp, format='pdf', bbox_inches="tight") #
            pp.close()
            plt.close()