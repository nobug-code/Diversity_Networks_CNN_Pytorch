import visdom
from datetime import datetime

class Visdom():

    def __init__(self, args, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = args.save_dir
        self.vis = visdom.Visdom(env=self.env_name,port=8060)
        self.t_loss = None
        self.t_acc = None
        self.te_loss = None
        self.te_acc = None

    def train_loss(self, loss, step, var_name):
        self.t_loss = self.vis.line([loss], [step], win=self.t_loss,
                update='append' if self.t_loss else None, 
                opts=dict(
                    xlabel = 'Epoc',
                    ylabel = 'Loss',
                    title='Loss'
                    )
                )
    def train_acc(self, acc, step, var_name):
        self.t_acc = self.vis.line([acc], [step], win=self.t_acc,
                update='append' if self.t_acc else None, 
                opts=dict(
                    xlabel = 'epoc',
                    ylabel = 'ACC',
                    title='ACC'
                    ))
    def test_loss(self, loss, step, var_name):
        self.te_loss = self.vis.line([loss], [step], win=self.te_loss,
                update='append' if self.te_loss else None, 
                opts=dict(
                    color = 'red',
                    xlabel = 'Epoc',
                    ylabel = 'test_Loss',
                    title='Test_Loss'
                    )
                )
    def test_acc(self, acc, step, var_name):
        self.te_acc = self.vis.line([acc], [step], win=self.te_acc,
                update='append' if self.te_acc else None, 
                opts=dict(
                    color = 'red',
                    xlabel = 'Epoc',
                    ylabel = 'test_ACC',
                    title='Test_ACC'
                    ))
