import numpy as np
import math

def vector_normalizer(vector, order = 7, anchor = 0):
	norm = np.float64([0])

	vector = np.float64(vector)
	vector = np.float64(vector / np.sum(vector))

	output = np.float64(np.zeros([len(vector)]))

	for i in range(len(vector)):
		v = np.float64([round(vector[i], order)])
		#v = np.float64([vector[i]])
		
		if v[0] != 0.0:
		#	print "current norm", norm[0]
			if (norm + v)[0] < 1.0:
				#print "add element ", v
				output[i] = v[0]
				norm = v + norm
				norm = np.round(norm, order)
			else:
				#print "next element is too large"
				v = np.float64([1.0]) - norm
				#print "add last one ", v
				output[i] = round(v[0], order)
				norm = v + norm
				norm = np.round(norm, order)
	while norm[0] < 1.0:
		#print "current norm[0]", norm[0], " still < 1.0"
		p = np.float64([1.0]) - norm
		p = np.round(p, order)
		output[anchor] = (np.float64([output[anchor]]) + p)[0]
		output[anchor] = round(output[anchor], order)
		norm = norm + p
		norm = np.round(norm, order)
		#print "add ", p, " to element ", anchor, " making it ", output[anchor]

	#print norm
	#norm = np.float64([0])
	#for i in range(len(output)):
	#	norm = norm + np.float64([output[i]])

	'''
	if norm[0] < 1.0:
		file = open('error', 'w')
		print "Fuck"
		#print "norm[0]< 1.0"
		for i in output:
			if i != 0.0:
				print i
			file.write(str(i) + ' ')
		return None
	elif norm[0] > 1.0:
		file = open('error', 'w')
		print "FUUUUUUUUUUUUUUUUUCK"
		#print "norm[0]> 1.0"
		#for i in output:
		#	if i != 0.0:
		#		print i
		#	file.write(str(i) + ' ')
		return None
	elif norm[0] == 1.0:
	
		print "Yeah!!"
		return output
	'''
	return output

if __name__ == "__main__":
	file = open('test', 'r')
	lines = file.readlines()
	line = lines[0].split(' ')
	test = []
	for i in line:
		 if i != '':
			test.append(float(i))

	#test = np.random.randint(100, size = 10).astype(float)
	#test = np.array([ 61.,  48.,  89.,  33.,  50.,  48.,  29.,   2.,  77.,  82.])
	#print test
	vector_normalizer(test, 7)
