# IMPORT GRAPH

abstraction:

        tree <- partition
        [local: statistic, plot]

    	statistic <- tree

    	plot <- tree, partition, recursive.recursive_tree

---

boltzmann:

        boltzmann_partition <- abstraction.partition

---

recursive:

        recursive_tree <- abstraction.tree, abstraction.partition,
        abstraction.helpers
        [local: <- conversions, transformations, random.random_trees]

    	conversions <- recursive_tree, rooted.rooted_tree

    	transformations <- recursive_tree, abstraction.helpers

---

rooted:

        rooted_tree <- abstraction.tree, abstraction.partition
        [local: <- conversions]

    	conversions <- recursive.recursive_tree, rooted_tree

---

containers:

        trees <- recursive.recursive_tree, recursive.conversions,
        rooted.rooted_tree, rooted.conversions, boltzmann.boltzmann_tree

    	partitions <- abstraction.partitions, boltzmann.boltzmann_partition

---

random:

        random_trees <- abstraction.tree, abstraction.helpers,
        recursive.recursive_tree, rooted.rooted_tree,
        containers.trees, boltzmann.boltzmann_tree,
        tree_generators, tree_probabilities, crump_jagers_mode

        tree_probabilities <- abstraction.tree, recursive.recursive_tree,
        rooted.rooted_tree, random_partitions, containers.trees,

        tree_generators <- recursive.recursive_tree, crump_jagers_mode,
        random_partitions

    	random_partitions <- abstraction.helpers, abstraction.partition,
        containers.partitions, boltzmann.boltzmann_partition

    	crump_jagers_mode <- recursive.recursive_tree, point_processes
    	[local: <- random.trees]

---

tests:

        tests <- containers.partitions, containers.trees,
        abstraction.partition, recursive.recursive_tree,
        random.random_trees, random.random_partitions
