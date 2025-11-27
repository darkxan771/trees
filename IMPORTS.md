# IMPORT GRAPH

abstraction: 	tree <- partition
				[local: statistic, plot]

				statistic <- tree

				plot <- tree, partition, recursive.recursive_tree

_______________

boltzmann: 		boltzmann_partition <- abstraction.partition


_______________

recursive:		recursive_tree <- abstraction.tree, abstraction.partition,
				abstraction.helpers
				[local: <- conversions, transformations, random.random_trees]

				conversions <- recursive_tree, rooted.rooted_tree

				transformations <- recursive_tree, abstraction.helpers

_______________

rooted:			rooted_tree <- abstraction.tree, abstraction.partition
				[local: <- conversions]

				conversions <- recursive.recursive_tree, rooted_tree

_______________

containers:		trees <- recursive.recursive_tree, recursive.conversions,
	 			rooted.rooted_tree, rooted.conversions, boltzmann.boltzmann_tree

				partitions <- abstraction.partitions, 
				boltzmann.boltzmann_partition

_______________

random:         random_trees <- recursive.recursive_tree, rooted.rooted_tree,
                containers.trees, boltzmann.boltzmann_tree, 
                probabilities_and_generators, crump_jagers_mode

                probabilities_and_generators <- recursive.recursive_tree, rooted.rooted_tree, random_partitions, containers.trees,
                crump_jagers_mode

				random_partitions <- abstraction.helpers, 
				abstraction.partition, 
                containers.partitions, boltzmann.boltzmann_partition

				crump_jagers_mode <- recursive.recursive_tree, point_processes
				[local: <- random.trees]
_______________

tests:			tests <- containers.partitions, containers.trees, 
				abstraction.partition, recursive.recursive_tree, 
				random.random_trees, random.random_partitions


