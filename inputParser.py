import argparse

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--lrate', action='store', dest='learning_rate',help='Learning Rate, default = 0.1',type=float,default = 0.1)

	parser.add_argument('--lr_decay', action='store', dest='l_decay',help='Learning Rate decay, default = 1',type=float,default = 1)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 200',type=int,default = 200)

	parser.add_argument('--b_size', action='store', dest='batch_size',help='batch size, default = 500',type=int,default = 500)

	parser.add_argument('--k_size0', action='store', dest='nkerns0',help='first kern size, default = 20',type=int,default = 20)

	parser.add_argument('--k_size1', action='store', dest='nkerns1',help='second kern size, default = 50',type=int,default = 50)

	parser.add_argument('--filter', action='store', dest='filtering',help='filtering, default = 5',type=int,default = 5)

	parser.add_argument('--pool', action='store', dest='pool_size',help='pool size, default = 2',type=int,default = 2)

	parser.add_argument('--channel', action='store', dest='channel',help='channel, for grayscale 1, for RGB 3, default = 1',type=int,default = 1)

	parser.add_argument('--height', action='store', dest='height',help='image height, default = 28',type=int,default = 28)
	parser.add_argument('--width', action='store', dest='width',help='image width, default = 28',type=int,default = 28)

	parser.add_argument('--hidden', action='store', dest='hidden_size',help='hidden size of softmax layer, default = 500',type=int,default = 500)

	parser.add_argument('--benchmark', action='store', dest='benchmark',help='benchmark name, default = mnist.pkl.gz',default = 'mnist.pkl.gz')

	parser.add_argument('--denoiser-ae', action='store', dest='da_file',help='denoiser autoencoder file name')
	return parser



def get_parser_AE():
	parser = argparse.ArgumentParser()
	parser.add_argument('--lrate', action='store', dest='learning_rate',help='Learning Rate, default = 0.1',type=float,default = 0.1)
	parser.add_argument('--lr_decay', action='store', dest='l_decay',help='Learning Rate decay, default = 1',type=float,default = 1)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 50',type=int,default = 50)

	parser.add_argument('--v_size', action='store', dest='visible_size',help='visible size of input, default = 784',type=int,default = 784)

	parser.add_argument('--hidden', action='store', dest='hidden_size',help='hidden layer size, default = 500',type=int,default = 1000)

	parser.add_argument('--c_rate', action='store', dest='corruption_rate',help='corruption rate, default = 0.3',type=float,default = 0.3)

	parser.add_argument('--output-name', action='store', dest='fname',help='output filename, default = da-1000-030.pkl.out',default = "da-1000-030.pkl.out")

	parser.add_argument('--benchmark', action='store', dest='benchmark',help='benchmark name, default = mnist.pkl.gz',default = 'mnist.pkl.gz')

	return parser
