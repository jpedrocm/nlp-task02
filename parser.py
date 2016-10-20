import random
from nltk import induce_pcfg, Nonterminal
from nltk.corpus import treebank

def create_sets():
	train_set = []
	test_set = []

	full_raw_set = treebank.parsed_sents()

	full_set = map(lambda tree: binarize_tree(tree), full_raw_set)

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

def extract_rules(sentences):
	grammar_rules = []

	for sentence in sentences:
		grammar_rules += sentence.productions()

	return grammar_rules

def create_pcfg(rules):
	S = Nonterminal('S')
	return induce_pcfg(S, rules)

def main():
	train_set, test_set = create_sets()

	grammar_rules = extract_rules(train_set)

	grammar = create_pcfg(grammar_rules)

main()