# Source code of the Focused ACO version with Adaptive Parameter Tuning

To compile the code a C++17 compatible compiler is needed, however, we have only tested
GNU G++ v12.

To compile the code run: `make`.

After a successful compilation the program can be run from command line, e.g.:

    ./faco --alg=faco_apt --problem tsp/rl5915.tsp --mab=exp3


The algorithm is described in more details in a paper:

    @InProceedings{10.1007/978-3-031-70816-9_4,
        author="Skinderowicz, Rafa{\l}",
        editor="Nguyen, Ngoc Thanh
        and Franczyk, Bogdan
        and Ludwig, Andr{\'e}
        and N{\'u}{\~{n}}ez, Manuel
        and Treur, Jan
        and Vossen, Gottfried
        and Kozierkiewicz, Adrianna",
        title="Enhancing Focused Ant Colony Optimization for Large-Scale Traveling Salesman Problems Through Adaptive Parameter Tuning",
        booktitle="Computational Collective Intelligence",
        year="2024",
        publisher="Springer Nature Switzerland",
        address="Cham",
        pages="41--54",
        isbn="978-3-031-70816-9"
    }
