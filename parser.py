import random
from nltk import induce_pcfg, Nonterminal
from nltk.corpus import treebank

def create_sets():
	train_set = []
	test_set = []

	raw_full_set = treebank.parsed_sents()

	full_set = map(lambda tree: binarize_tree(tree), raw_full_set)

	full_set_size = len(full_set)
	full_set_indexes = range(full_set_size)
	train_set_indexes = random.sample(full_set_indexes, int(0.75*full_set_size))
	test_set_indexes = list(set(full_set_indexes) - set(train_set_indexes))

	train_set = map(lambda i: full_set[i], train_set_indexes)
	test_set = map(lambda i: full_set[i], test_set_indexes)

	return (train_set, test_set)

def binarize_tree(tree):
	tree.collapse_unary(collapsePOS = True, collapseRoot = True)
	tree.chomsky_normal_form()
	return tree

def extract_rules(trees):
	rules = {}
	for tree in trees:
		tree_rules = tree.productions()
		for rule in tree_rules:
			left_side = rule.lhs()
			right_side = rule.rhs()
			if left_side not in rules:
				rules[left_side] = {right_side: 1}
			else:
				if right_side not in rules[left_side]:
					rules[left_side][right_side] = 1
				else:
					rules[left_side][right_side] += 1
	return rules

def normalize_and_transform_rules(rules):
	grammar = {}
	for (left_side, right_sides_list) in rules.iteritems():
		total_instances = float(sum(right_sides_list.values()))
		for (right_side, num_of_instances) in right_sides_list.iteritems():
			if right_side not in grammar:
				grammar[right_side] = {left_side: (num_of_instances/total_instances)}
			else:
				if left_side not in grammar[right_side]:
					grammar[right_side][left_side] = (num_of_instances/total_instances)
				else:
					print "ERROR"
	return grammar

def create_pcfg(trees):
	raw_rules = extract_rules(trees)
	grammar = normalize_and_transform_rules(raw_rules)
	return grammar

def calculate_metric_of_sentence(labeled, predicted):
	num_of_same_label = len(filter(lambda x: x in labeled, predicted))

	precision = num_of_same_label / float(len(predicted))
	recall = num_of_same_label / float(len(labeled))
	f1 = precision / recall
	tagging_accuracy = 0 ############# TODO
	return [precision, recall, f1, tagging_accuracy]

def calculate_parser_metrics(list_of_sentence_metrics):
	parser_metrics_sum = {}
	num_of_sentences = len(List_of_sentence_metrics)

	parser_metrics_sum = [sum(x) for x in zip(*list_of_sentence_metrics)]

	parser_metrics = map(lambda x: x/num_of_sentences)

	return parser_metrics

def print_metrics(parser_metrics):
	print "METRICS"
	print "Precision: " + str(parser_metrics[0])
	print "Recall: " + str(parser_metrics[1])
	print "F-measure: " + str(parser_metrics[2])
	print "Tagging accuracy: " + str(parser_metrics[3])

def main():
	train_set, test_set = create_sets()

	pcfg = create_pcfg(train_set)

main()