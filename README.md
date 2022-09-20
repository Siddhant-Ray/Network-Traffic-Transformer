#  Network Traffic Transformer (NTT)

This work was undertaken as part of my master thesis at ETH Zurich, from Feb 2022 to Aug 2022, titled `Advancing packet-level traffic predictions with Transformers`. We present a new transformer-based architecture, to learn network dynamics from packet traces.

We design a `pre-training` phase, where we learn fundamental network dynamics. Following this, we have a `fine-tuning` phase, on different network tasks, and demonstrate that pre-training well leads to generalization to multiple fine-tuning tasks.

### Original proposal: 
* [`Project proposal`](https://nsg.ee.ethz.ch/fileadmin/user_upload/thesis_proposal_packet_transformer.pdf)

### Supervisors: 
* [`Alexander Dietmüller`](https://nsg.ee.ethz.ch/people/alexander-dietmueller/)
* [`Dr. Romain Jacob`](https://nsg.ee.ethz.ch/people/romain-jacob/)
* [`Prof. Dr. Laurent Vanbever`](https://nsg.ee.ethz.ch/people/laurent-vanbever/)

### Research Lab: 
* [`Networked Systems Group, ETH Zurich`](https://nsg.ee.ethz.ch/home/)

### We redirect you to the following sections for further details.

* [`Code and reproducing instructions:`](workspace/README.md)

* [`Thesis TeX and PDF files`](report/)

* [`Literature files`](literature/)

* [`Slides TeX and PDF files`](presentation/)

### NOTE 1:
The experiments conducted in this project are very involved. Understanding and reproducing them from just the code and comments alone will be quite hard, inspite of the instructions mentioned in the given [`README`](workspace/README.md). For more detailed understanding, we invite you to read the thesis ([`direct link`](report/thesis.pdf)). You can also check out an overview on the presentation slides ([`direct link`](presentation/slides.pdf))

For any further questions or to discuss related research ideas, please feel free to contact me by [`email.`](mailto:siddhant.r98@gmail.com)

### NOTE 2:
Some results from the thesis have been written as a paper titled ```A new hope for network model generalization``` and the same has been accepted for presentation at [ACM HotNets 2022](https://conferences.sigcomm.org/hotnets/2022/). More details will follow and a link to the paper will be made available soon.

### NOTE 3: 
The thesis has now been published under the ETH Research collection, which is open access. It can be accessed from here ([`direct link`](https://www.research-collection.ethz.ch/handle/20.500.11850/569234))

