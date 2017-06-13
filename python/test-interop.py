#!/usr/bin/python

print('hello from test-interop.py script')

import argparse
parser = argparse.ArgumentParser(description='test-interop')
parser.add_argument('--foo', action='store_true',help='foo flag')
parser.add_argument('--bar', action='store_true',help='bar flag')
args = parser.parse_args()

print('foo: ' + str(args.foo))
print('bar: ' + str(args.bar))


def add( x, y ):
	print('inside function add()')
	return x + y

def mul( x, y ):
	print('inside function mul()')
	return x * y
