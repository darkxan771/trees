# TODO

programs:
- complete test_random()
- documentation integer partitions / other distributions (uniform) / method get_random_element


maths:
- expectation and limit law for the profile (at height alpha log n)
- law of a random cut (conditionnally to its size)

# IMPORT GRAPH

abstraction: 	tree <- partition
				statistic <- tree
				plot <- tree, recursive.recursive_tree

recursive:		recursive_tree <- abstraction.tree, abstraction.partition
				[local: <- conversions, transformations]
				conversions <- recursive_tree, rooted.rooted_tree
				transformations <- recursive_tree, abstraction.helpers

rooted:			rooted_tree <- abstraction.tree, abstraction.partition
				[local: <- conversions]
				conversions <- recursive.recursive_tree, rooted_tree

containers:		trees <- recursive.recursive_tree, recursive.conversions,
	 			rooted.rooted_tree, rooted.conversions, random.boltzmann
				[local: <- random.random_trees]
				partitions <- abstraction.partitions

random:         random_trees <- recursive.recursive_tree, rooted.rooted_tree 					boltzmann, containers.trees 
				random_partitions <- abstraction.helpers, abstraction.partition, containers.partitions
				crump_jagers_mode <- recursive.recursive_tree

tests:			tests <- containers.partitions, containers.trees, 
				abstraction.partition, recursive.recursive_tree, 
				random.random_trees














