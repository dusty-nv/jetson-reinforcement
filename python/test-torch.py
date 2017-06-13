#!/usr/bin/python

print('hello from test-torch.py script')
print('importing torch...')

import torch

print('import complete.')
print('cuda available: ' + str(torch.cuda.is_available()))
print('testing basic tensor operations')

a = torch.cuda.FloatTensor(2).zero_()
print(a)
b = torch.randn(2).cuda()
print(b)
c = a + b
print(c)


def foo( x ):
	print('tensor interop test')
	print(x)
	#return x.dim()
	#return x.size()[0]
	#return x.sum()
	return x.std()


