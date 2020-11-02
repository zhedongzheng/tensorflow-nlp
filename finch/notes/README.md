* Transformer

	* Advantages

		* It make no assumptions about the temporal/spatial relationships across the data. This is ideal for processing a set of objects (for example, StarCraft units).

		* Layer outputs can be calculated in parallel, instead of a series like an RNN.

		* Distant items can affect each other's output without passing through many RNN-steps, or convolution layers (see Scene Memory Transformer for example).

		* It can learn long-range dependencies. This is a challenge in many sequence tasks.

	* Downsides

		* For a time-series, the output for a time-step is calculated from the entire history instead of only the inputs and current hidden-state. This may be less efficient.

		* If the input does have a temporal/spatial relationship, like text, some positional encoding must be added or the model will effectively see a bag of words.

	* Reference

		* [TF 2 Tutorial](https://www.tensorflow.org/alpha/tutorials/text/transformer)
