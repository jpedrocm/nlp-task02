import random
from nltk import Nonterminal, Tree
from nltk.corpus import treebank
from nltk.tag.mapping import map_tag

def create_sets():
	train_set = []
	test_set = []

	raw_full_set = treebank.parsed_sents()

	full_set = map(lambda tree: transform_tree(tree), raw_full_set)

	full_set_size = len(full_set)
	full_set_indexes = range(full_set_size)
	train_set_indexes = random.sample(full_set_indexes, int(0.75*full_set_size))
	test_set_indexes = list(set(full_set_indexes) - set(train_set_indexes))

	train_set = map(lambda i: full_set[i], train_set_indexes)
	test_set = map(lambda i: full_set[i], test_set_indexes)

	return (train_set, test_set)

def transform_tree(tree):
	new_tree = Tree.fromstring("(NEW_ROOT" + str(tree) + ")")
	new_tree.collapse_unary()
	new_tree.chomsky_normal_form()
	for leaf in new_tree.subtrees(lambda t: t.height()==2):
		leaf.set_label(map_tag('en-ptb', 'universal', leaf.label()))
	return new_tree

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
	rules[Nonterminal("NOUN")] = {("UNK",): 1}
	return rules

def normalize_and_transform_rules(rules):
	grammar = {}
	for (left_side, right_sides_list) in rules.iteritems():
		total_instances = float(sum(right_sides_list.values()))

		for (right_side, num_of_instances) in right_sides_list.iteritems():
			probability = num_of_instances / total_instances
			
			grammar = add_rule_to_grammar(grammar, left_side, right_side, probability)

	return grammar

def add_rule_to_grammar(cur_grammar, left_side, right_side, probability):
	if right_side not in cur_grammar:
		cur_grammar[right_side] = {left_side: probability}
	else:
		if left_side not in cur_grammar[right_side]:
			cur_grammar[right_side][left_side] = probability
		else:
			raise NameError("GRAMMAR CONSTRUCTION ERROR")

	return cur_grammar

def create_pcfg(trees):
	raw_rules = extract_rules(trees)
	grammar = normalize_and_transform_rules(raw_rules)
	return grammar

def extract_words_pos_tags(tree):
	tags = []
	word_tag_tuples = tree.pos()
	tags = map(lambda duple: duple[1], word_tag_tuples)
	return tags

def calculate_tagging_accuracy(candidate_tree, gold_tree):
	candidate_tags = extract_words_pos_tags(candidate_tree)
	gold_tags = extract_words_pos_tags(gold_tree)

	num_of_tags = len(gold_tags)

	num_of_equal_tags = len(filter(lambda v: v==True, [candidate_tags[i]==gold_tags[i] for i in range(0,num_of_tags)]))

	tagging_accuracy = num_of_equal_tags / float(num_of_tags)

	return tagging_accuracy

def extract_brackets(tree):
	brackets = []

	tree_leaves = tree.leaves()
	for subtree in tree.subtrees(lambda t: t.height() > 2):
		subtree_leaves = subtree.leaves()
		last_subtree_index = len(subtree_leaves)-1
		for i in range(0, len(tree_leaves)):
			if subtree_leaves[0]==tree_leaves[i] and subtree_leaves[last_subtree_index]==tree_leaves[i+last_subtree_index]:
				start_index = i
				end_index = start_index + last_subtree_index
				brackets.append((subtree.label(), start_index, end_index))
				break
	return brackets

def calculate_metric_of_sentence(candidate_tree, gold_tree):
	candidate_brackets = extract_brackets(candidate_tree)
	gold_brackets = extract_brackets(gold_tree)

	num_of_equal_brackets = len([bracket in gold_brackets for bracket in candidate_brackets])

	precision = num_of_equal_brackets / float(len(candidate_brackets))
	recall = num_of_equal_brackets / float(len(gold_brackets))

	f1 = precision / recall
	tagging_accuracy = calculate_tagging_accuracy(candidate_tree, gold_tree)

	return [precision, recall, f1, tagging_accuracy]

def calculate_parser_metrics(list_of_sentence_metrics):
	parser_metrics_sum = {}
	num_of_sentences = len(list_of_sentence_metrics)

	parser_metrics_sum = [sum(x) for x in zip(*list_of_sentence_metrics)]

	parser_metrics = map(lambda x: x/num_of_sentences)

	return parser_metrics

def print_metrics(parser_metrics):
	print "METRICS"
	print "Precision: " + str(parser_metrics[0])
	print "Recall: " + str(parser_metrics[1])
	print "F-measure: " + str(parser_metrics[2])
	print "Tagging accuracy: " + str(parser_metrics[3])

def cky(words, pcfg):
	score = [[{} for i in range(len(words)+1)] for i in range(len(words)+1)]
	back = [[{} for i in range(len(words)+1)] for i in range(len(words)+1)]
	i = 0

	keys = pcfg.keys()
	for w in words:
		tup = (w,)
		if(tup in keys):
			for a in list(pcfg[tup].keys()):
				score[i][i+1][a] = pcfg[tup][a]
		else:
			tup = ("UNK",)
			for a in list(pcfg[tup].keys()):
				score[i][i+1][a] = pcfg[tup][a]

		#Unarias Nao Terminais
		added = True

		while(added):
			added = False
			bs = score[i][i+1].keys()
			for b in bs:
				tup_b = (b,)
				if((tup_b in pcfg) and score[i][i+1][b]>0):
					for a in list(pcfg[tup_b].keys()):
						if((a not in score[i][i+1])):
							score[i][i+1][a] = 0
						prob = pcfg[tup_b][a]*score[i][i+1][b]
						if(score[i][i+1][a]<prob):
							score[i][i+1][a] = prob
							back[i][i+1][a] = b
							added = True
		
		i = i+1

	for span in range(2,len(words)):
		for begin in range(len(words)-span):
			end = begin + span
			for split in  range(begin+1, end-1):
				bs = score[begin][split].keys();
				cs = score[split][end].keys();

				for b in bs:
					for c in cs:
						tup = (b,c)
						for a in list(pcfg[tup].keys()):
							if((a not in score[begin][end])):
								score[begin][end][a] = 0

							prob = 	score[begin][split][b]*score[split][end][c]*pcfg[tup][a]
							if(prob>score[begin][end][a]):
								score[begin][end][a] = prob
								back[begin][end][a] = tup

							added = True

							while(added):
								added = False
								bsu = score[begin][end].keys()
								for bu in bsu:
									tup_bu = (bu,)
									if((tup_bu in pcfg) and score[begin][end][bu]>0):
										for au in list(pcfg[tup_bu].keys()):
											if((au not in score[begin][end])):
												score[begin][end][au] = 0
											prob = pcfg[tup_bu][au]*score[begin][end][bu]
											if(score[begin][end][au]<prob):
												score[begin][end][au] = prob
												back[begin][end][au] = bu
												added = True								

	return build_candidate_tree(score, back)

def build_candidate_tree(score, back):
	return 0

def process_pcfg(pcfg):
	transform_pcfg = {}
	for key in pcfg.keys():
		transform_pcfg[key[0]] = pcfg[key]

	return transform_pcfg

def main():
	train_set, test_set = create_sets()

	pcfg = create_pcfg(train_set)

	list_of_sentence_metrics = []

	i = 0
	for gold_tree in test_set:
		if(i == 0):
			words = gold_tree.leaves()
			candidate_tree = cky(words,pcfg)
			i = i+1
		#list_of_sentence_metrics.append(calculate_metric_of_sentence(candidate_tree, gold_tree))"""

	#print_metrics(calculate_parser_metrics(list_of_sentence_metrics))

main()