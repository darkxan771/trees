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
	 			rooted.rooted_tree, rooted.conversions, random.boltzmann_tree
				[local: <- random.random_trees]
				partitions <- abstraction.partitions, random.random_partitions

random:         random_trees <- recursive.recursive_tree, rooted.rooted_tree 					boltzmann_tree, containers.trees 
				random_partitions <- abstraction.helpers, abstraction.partition, containers.partitions,
				boltzmann_partition
				crump_jagers_mode <- recursive.recursive_tree

tests:			tests <- containers.partitions, containers.trees, 
				abstraction.partition, recursive.recursive_tree, 
				random.random_trees, random.random_partitions