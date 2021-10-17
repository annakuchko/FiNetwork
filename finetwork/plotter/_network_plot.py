# TODO: plot criterion values kde
import seaborn as sns
import matplotlib.pyplot as plt

class NetPlot:
    
    def __init__(self, criterion_values):
        self.criterion_values = criterion_values
        
    def _plot_criterion_pdf(self):
            criterion_values = self.criterion_values
            sns.displot(data=criterion_values, 
                        rug=True,
                        kind="kde",
                        )
