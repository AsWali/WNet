import matplotlib.pyplot as plt
import numpy as np

class PlotLosses:

    def __init__(self, n_cut_file, rec_losses_file) -> None:
        self.n_cut_file = n_cut_file
        self.rec_losses = rec_losses_file
        super().__init__()

    def print_and_plot(self):
        n_cut_losses = np.load(self.n_cut_file)
        rec_losses = np.load(self.rec_losses)
        print(n_cut_losses)
        print(rec_losses)
        plt.plot(n_cut_losses)
        plt.show()
        plt.plot(rec_losses)
        plt.show()

p = PlotLosses('losses_output/1.npy','losses_output/2.npy')
p.print_and_plot()