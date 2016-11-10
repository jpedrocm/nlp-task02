import random, time
from nltk import Nonterminal, Tree
from nltk.corpus import treebank
from nltk.tag.mapping import map_tag

num_of_no_trees = 0

def create_sets():
	train_set = []
	test_set = []

	raw_full_set = treebank.parsed_sents()
	filtered_full_set = filter_set_from_none(raw_full_set)

	final_set = map(lambda tree: transform_tree(tree), filtered_full_set)

	final_set_size = len(final_set)
	final_set_indexes = range(final_set_size)
	train_set_indexes = random.sample(final_set_indexes, int(0.75*final_set_size))
	test_set_indexes = list(set(final_set_indexes) - set(train_set_indexes))

	train_set = map(lambda i: final_set[i], train_set_indexes)
	test_set = map(lambda i: final_set[i], test_set_indexes)

	return (train_set, test_set)

def filter_set_from_none(full_set):
	return filter(lambda tree: not contain_none(tree), full_set)

def contain_none(tree):
	return '-NONE-' in map(lambda (w,t): t, tree.pos())

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
	rules[Nonterminal("NOUN")][("UNK",)] = 1
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
	global num_of_no_trees
	if candidate_tree is None:
		num_of_no_trees+=1
		return [0.0, 0.0, 0.0, 0.0]
	candidate_brackets = extract_brackets(candidate_tree)
	gold_brackets = extract_brackets(gold_tree)

	equal_brackets = [g_bracket for g_bracket in gold_brackets for c_bracket in candidate_brackets if g_bracket==c_bracket]
	num_of_equal_brackets = len(equal_brackets)

	precision = num_of_equal_brackets / float(len(candidate_brackets))
	recall = num_of_equal_brackets / float(len(gold_brackets))

	f1 = 2*precision*recall / (precision+recall)
	tagging_accuracy = calculate_tagging_accuracy(candidate_tree, gold_tree)

	metric = [precision, recall, f1, tagging_accuracy]
	return metric

def calculate_parser_metrics(list_of_sentence_metrics):
	parser_metrics_sums = {}
	num_of_sentences = len(list_of_sentence_metrics)

	parser_metrics_sums = [sum(x) for x in zip(*list_of_sentence_metrics)]

	parser_metrics = map(lambda x: x/num_of_sentences, parser_metrics_sums)
	return parser_metrics

def print_metrics(parser_metrics):
	print "METRICS"
	print "Precision: " + str(parser_metrics[0])
	print "Recall: " + str(parser_metrics[1])
	print "F-measure: " + str(parser_metrics[2])
	print "Tagging accuracy: " + str(parser_metrics[3])

def cky(words, pcfg):
	words_size = len(words)

	score = [[{} for i in range(words_size+1)] for j in range(words_size+1)]
	back = [[{} for i in range(words_size+1)] for j in range(words_size+1)]
	i = 0

	keys = pcfg.keys()
	for w in words:
		tup = (w,)
		if tup in keys:
			for a in pcfg[tup].keys():
				score[i][i+1][a] = pcfg[tup][a]
		else:
			tup = ("UNK",)
			for a in pcfg[tup].keys():
				score[i][i+1][a] = pcfg[tup][a]

		score[i][i+1], back[i][i+1] = create_unarias(score[i][i+1], back[i][i+1], pcfg)		
		i = i+1

	for span in range(2, words_size+1):
		for begin in range(words_size-span+1):
			end = begin + span

			for split in range(begin+1,end):
				bs = score[begin][split].keys()
				cs = score[split][end].keys()

				possible_duples = [(bu,cu) for bu in bs for cu in cs]
				
				for tup in possible_duples:
					if tup in pcfg:
						bu = tup[0]
						cu = tup[1]
						for au in pcfg[tup].keys():
							if au not in score[begin][end]:
								score[begin][end][au] = 0.0

							prob = score[begin][split][bu]*score[split][end][cu]*pcfg[tup][au]
							if prob>score[begin][end][au]:
								score[begin][end][au] = prob
								back[begin][end][au] = (split,bu,cu)
							score[begin][end], back[begin][end] = create_unarias(score[begin][end], back[begin][end], pcfg)
	return build_candidate_tree(score, back, words)

def create_unarias(cell, back_cell, pcfg):
	added = True
	while(added):
		added = False
		bsu = cell.keys()
		for bu in bsu:
			tup_bu = (bu,)
			if tup_bu in pcfg and cell[bu]>0:
				for au in pcfg[tup_bu]:
					if au not in cell:
						cell[au] = 0
					prob = pcfg[tup_bu][au]*cell[bu]
						
					if cell[au]<prob:
						cell[au] = prob
						back_cell[au] = bu		
					
						added = True
	return cell, back_cell

def build_candidate_tree(score, back, words):
	li = 0
	ri = len(words)
	tagi = Nonterminal('NEW_ROOT')
	if tagi not in back[li][ri]:
		return None
	tree_string = '(' + str(tagi) + ' ' + build_tree(back, li, ri, tagi, words, "")
	candidate_tree = Tree.fromstring(tree_string)
	return candidate_tree

def build_tree(back, li, ri, tagi, words, cur):
	if abs(ri-li)==1:
		return cur + words[li] +') '

	backT = back[li][ri][tagi]
	if isinstance(backT, Nonterminal):
		tagi = backT
		cur += '\n(' + str(tagi) + ' '
		cur = build_tree(back, li, ri, tagi, words, cur)
	else:
		split = backT[0]
		cur+= '\n(' + str(backT[1]) + ' '
		cur = build_tree(back, li, split, backT[1], words, cur)
		cur += '\n(' + str(backT[2]) + ' '
		cur = build_tree(back, split, ri, backT[2], words, cur)
	return cur + ')'

def main():
	start_time = time.time()
	train_set, test_set = create_sets()

	pcfg = create_pcfg(train_set)

	list_of_sentence_metrics = []

	for gold_tree in test_set:
		words = gold_tree.leaves()
		candidate_tree = cky(words, pcfg)
		list_of_sentence_metrics.append(calculate_metric_of_sentence(candidate_tree, gold_tree))
	print_metrics(calculate_parser_metrics(list_of_sentence_metrics))
	print "Total of sentences: " + str(len(test_set))
	print "Sentences not parsed: " +str(num_of_no_trees)
	print "Time spent: " + str(time.time()-start_time)

main()