Papers
Structured layers
    Displacement rank:
    - Sindhwani NIPS
    - other displacement rank?
        - they used Toeplitz-like for kernel approximation?
        - long paper on textures
    - other "structured efficient linear layers": fast JLT (Ailon, Chazelle), FastFood, HD^3 from choromonski?
    - include ACDC, mention that our class covers that
Compression/Pruning/Hashing (posthoc)
    Song Han
    HashedNets
    MobileNets?
Theory
    (ICML17) Pan: Displacement rank for neural nets


#Outline
1. Intro (background/related work, what we do)
1.1 Technical Background on LDR (interleave with other parts of intro?)
1.2 Displacement Operators as representing invariances / data-distribution/dependent blah...
2. Classes we consider, our contributions, experiments
3. [experiments], conclusions/future work



###Intro
- Standard pitch on compressing full layers of neural networks
    * Emphasis on memory (model size): cut down parameters
    * mention broad types of approaches (pruning/sparsity, vs. structured)
        * Within structured class, Mention relaxation from convolution to toeplitz/circulant to toeplitz-like
        * for other structures, mention line of work (involving mixed diagonal and hadamard/fourier) for ACDC/[Cheng et al.]  and how this is included in our class?
- In this work, we consider displacement. Why?
    * standard reasons for structured matrices over pruning/sparsity/adhoc means of parameter-sharing:
        * strong constraint on the structure of the matrix / Impose a strong regulation on the learned representation, whereas hashing/sparsity etc. are heuristic/adhoc
        * guaranteed compression instead of heuristics
        * can learn over the strong structure directly instead of multiple passes for projection/pruning/approximation
        * results in dense structure on the final weight matrix instead of sparse or repeated parameters

    * beyond this, a goal of this work is to be able to represent structure in the weight matrices _that reflects structure in the data_, whereas past approaches - particularly the heuristic pruning approaches but even the dense-structured approaches - assume the _unconstrained_ representations are optimal and merely use structure as a heuristic means of compression and acceleration.
        - we find places where compressed models perform better than unconstrained (due to regularization?)

- Towards this goal: We focus on the displacement approach, which has an interpretation of representing certain types of matrix structures. We consider a much broader family of displacement structures that unify and extend many of the previously considered _structured_ classes (even some that were not viewed as displacement-structured) and achieve improved performance on downstream tasks, while also seeking to explain what makes this family of constructions effective.

\todo{perhaps move the below para to 1.2}
Our class of more general structured layers uses a more general form of displacement structures, with additional parameters that we interpret as governing the structure of the weight matrix or invariances in the data. We show that keeping a fixed parameter budget, increasing the number of parameters for structure results in performance increase.
Using these more general parametrizations, we explore how underlying parameters affect and govern the structure of the weight matrix [invariances of data], and how parameters can be utilized most effectively when keeping a fixed parameter budget.

### Technical background

### 1.2 (Intro after introducing DR)
        - A mystery not mentioned by Sindhwani: the displacement equation is actually just a linear map on the space of matrices, so the space of LDR is isomorphic to the space of low rank. Why does e.g. Toeplitz-like perform better?
        - interpret displacement equation as a part consisting of overall model/invariance, and a more specific portion for fine-tuning. So for toeplitz-like, DR consists of specifying a low rank model within a class that has certain invariances/structure already.
    * Furthermore, we point out connections between the displacement approach and learning invariances in data, a [benefit] that has been seldom mentioned even in the works considering low displacement rank. Just as how convolutions/max-pooling can be viewed as being resistant to local transformations...
    * We significantly extend the range of structures considered. Using the connection between Toeplitz-like matrices and shift invariances as an example [wc], we consider learning the displacement operators (the part of the model that governs structure) directly. We perform a comparison between the _types_ of parameters, showing that on a _per parameter basis, the parameters governing structure are much more effective at increasing performance than parameters for fine-tuning_

### Technical / Classes we consider
* Mention that forward multiplication is also fast (asymptotically)
* hence backprop automatically fast. footnote: Mention that Sindhwani and Pan ICML17 papers provide gradient algorithm even in the restricted setting of fixed displacement operators, but we're not sure why

* how does displacement connect to invariance? show equation: $ZA-AZ=R, Z(Ax) - Rx = A(Zx)$

* First class: circulant sparsity pattern.
    - Straightforward extension to Toeplitz-like matrices (mention in intro how toeplitz connects to convolutions), where we can observe the effects of learning over the operator
    - Both NIPS15 and ICML17 consider a certain type of displacement for technical purposes, but don't provide examples of reasonable structures satisfying it. This class is one of the most basic forms and satisfies the most basic conditions for efficient recovery/multiplication.
    - It turns out that this class includes other structured classes that were not understood through the lens of displacement rank. This emphasizes the representative power of DR, while this work also provides insights into such previous work.
* Second class: tridiagonal
    - It was only recently observed that much more general classes of displacement structures admit fast (near-linear) matrix multiplication.
    - This class includes all of the conventional classes of displacement structure, thus can learn between any of them


### Empirical Results
- why do we use the particular frameworks we test? (single layer, simple dataset)
    * simple to control for stuff and make objective comparisons
    * in the same vein as previous works (hashednets, sindhwani) so easy to comparable to the other structures tried


###Further work
quasiseparable operators
understand how they interact with optimization algorithms: distinct structures yield strange distinct learning curves


