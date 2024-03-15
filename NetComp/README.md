# NetComp v0.2

[![Travis CI](https://travis-ci.org/peterewills/NetComp.svg?branch=master)](https://travis-ci.org/peterewills/NetComp)
[![codecov](https://codecov.io/gh/peterewills/NetComp/branch/master/graph/badge.svg)](https://codecov.io/gh/peterewills/NetComp)
 
NetComp is a Python library implementing various algorithms for comparison of networks. A network is a collection of nodes and edges, also known as a graph.
 
## Installation
 
To install NetComp, do

	pip install netcomp
 
## Usage

Although many common packages, such as [NetworkX](https://github.com/networkx/networkx) treat graphs as fundamental objects, this package treats the adjacency matrix as the object of interest. In this usage example, we will use NetworkX to generate our graphs, then transform them to adjacency matrices before using NetComp

	>> import networkx as nx
	>> import netcomp as nc
	>> G1,G2 = [nx.erdos_renyi_graph(10,1/2) for _ in range(2)] # 2 random graphs
	>> A1,A2 = [nx.adjacency_matrix(G) for G in [G1,G2]]
	>> nc.lambda_dist(A1,A2,kind='laplacian',k=10)
 
## Contributing
 
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :+1:
 
## History

For more details on current status and upcoming improvements, see `STATUS.md`.
 
Version 0.1 (2017-09-25) - Slow algorithms in place.  
Version 0.2 (2017-10-13) - Sparse matrix support added.
 
## Credits
 
Author: Peter Wills (peter.e.wills@gmail.com)
 
## License
 
The MIT License (MIT)

Copyright (c) 2017 Peter Wills

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
