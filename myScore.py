#!/usr/bin/python
#
# scorer for Adam Meyers NLP class Spring 2016
# ver.1.0
#
# score a key file against a response file
# both should consist of lines of the form:   token \t tag
# sentences are separated by empty lines
#
import sys

def score (keyFileName, responseFileName):
	#KEY = bag of words model (im backwards)
	#Response = dev test file
	keyFile = open(keyFileName, 'r')
	key = keyFile.readlines()
	responseFile = open(responseFileName, 'r')
	response = responseFile.readlines()
	if len(key) != len(response):
    		print "length mismatch between key and submitted file"
		exit()
	correct = 0
	incorrect = 0
	keyGroupCount = 0
	keyStart = 0
	responseGroupCount = 0
	responseStart = 0
	correctGroupCount = 0
	# if(key[0] == 'result'):
	# 	key.remove(0)
	for i in range(len(key)):
		key[i] = key[i].rstrip('\n')
		response[i] = response[i].rstrip('\n')
		keyResult = key[i]
		responseFields = response[i].split('\t')
		responseText = responseFields[0]
		responseResult = responseFields[1]
		# print responseResult + keyResult
		if responseResult == keyResult:
			correct = correct + 1
		else:
			incorrect = incorrect + 1

	print correct, "out of", str(correct + incorrect) + " tags correct"
	accuracy = 100.0 * correct / (correct + incorrect)
	print "  accuracy: %5.2f" % accuracy

	# precision = 100.0 * correct / responseGroupCount
	# recall = 100.0 * correctGroupCount / keyGroupCount
	# F = 2 * precision  * recall / (precision + recall)
	# print "  precision: %5.2f" % precision
	# print "  recall:    %5.2f" % recall
	# print "  F1:        %5.2f" % F

def main(args):
	key_file = args[1]
	response_file = args[2]
	score(key_file,response_file)

if __name__ == '__main__': sys.exit(main(sys.argv))

## python score.chunk.py WSJ_24.pos-chunk response.chunk
