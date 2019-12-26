//! # Stackable Layers
//!
//! ## Construction
//! You generally start with specifying your OutputLayer onto which you then
//! [InnerLayer.push()](InnerLayer#method.push) inner layers.
//! the resulting layer is the input layer.
//!
//! ## Training
//!
//! You would then train the resulting network using [Backpropagaiton](Layer#method.backprop) by showing
//! it inputs and expected outputs.
//!
//!
//! ## Use
//! Once trained you can ask it to [calculate](Layer#tymethod.calculate) an output for any input.
//!
//! Layers, and therefore the whole Network, can be serialized and deserialized at any time, using
//! SerDe.
//!
//! ## Type Parameters
//! Layers take tons of type parameters, but you usually only have to give two, the activation
//! function and that layers size. those are also the leftmost type parameters. Other parameters
//! can be left as _.
//!
//! Only the input layer needs to have both its size and its inputs provided.
//!
//! Check out the example directory if you are still unsure.

use generic_array::{ArrayLength, GenericArray};

use rand::Rng;

use crate::activation::Activation;
use core::fmt::Debug;

use serde::{de::DeserializeOwned, Serialize};

pub trait AL = ArrayLength<f32> + Debug + Clone;
pub trait NL<Input: ArrayLength<f32>> =
	ArrayLength<GenericArray<f32, Input>> + ArrayLength<f32> + Debug + Clone;

fn loss<Len: AL>(errors: &GenericArray<f32, Len>) -> f32 {
	let sum: f32 = errors.iter().map(|e| e * e).sum();
	sum.sqrt()
}
/// A Layer takes a list of inputs, multiplexes and weights them onto its Neurons,
/// and produces a list of outputs, one for each neuron
///
/// Functions starting with an underscore are generally not to be called manually
// ToDo: split into two traits, one with a user facing interface and one with the implementation
// details
pub trait Layer<
	/// number of inputs this layer takes
	Input: AL,
	/// number of neurons this layer has
	Neurons: NL<Input>,
	/// number of neurons of the final layer
	FinalOut: AL,
	/// activation function
	A: Activation,
> :Debug+Clone+Serialize+DeserializeOwned {
	/// runs the inputs through this and lower layers, resulting in the output
	fn calculate(&self, inputs: &GenericArray<f32, Input>) -> GenericArray<f32, FinalOut>;

	/// reduces the inputs to a single input per neuron using the weights.
	/// i.e. calculates the input to the activation functions for the neurons
	fn weight(&self, inputs: &GenericArray<f32, Input>) -> GenericArray<f32, Neurons>;

	/// runs only this layers calculation, not recursing to deeper layers
	fn step(&self, weighted_inputs: &GenericArray<f32, Neurons>) -> GenericArray<f32, Neurons>;

	/// pass in the input and the expected output. calculates the error for each neuron
	/// and corrects itself
	/// speed should be 0..1 and modifies how strongly the weights are adjusted
	fn backprop(&mut self, input: &GenericArray<f32, Input>, correct_output: &GenericArray<f32, FinalOut>, speed: f32
		) -> (GenericArray<f32, Input>, GenericArray<f32, FinalOut>) {
		let weighted_inputs = self.weight(input);
		let output = self.step(&weighted_inputs);

		let (own_error, final_error) = self._get_error(output, correct_output, speed);

		let deltas = self._weight_errors(own_error, &weighted_inputs);
		let previous_errors = self._pre_error(&deltas);
		self._apply_deltas(deltas, &input, speed);

		// the errors are just passed up for informational purposes, so a training alg can
		// determine how wrong the network is without running an extra recognition step
		// its not actually used for backprop at all.
		(previous_errors, final_error)
	}


	/// gets this layers error, either by comparing with correct output or by calling lower layers
	fn _get_error(&mut self, output: GenericArray<f32, Neurons>, correct_output: &GenericArray<f32, FinalOut>, speed: f32
		) -> (GenericArray<f32, Neurons>, GenericArray<f32, FinalOut>);

	/// weights errors relative to activation. gets called by backprop, don't call this manually
	fn _weight_errors(&self, mut error: GenericArray<f32, Neurons>, weighted_inputs: &GenericArray<f32, Neurons>) -> GenericArray<f32, Neurons> {
		error.iter_mut().zip(weighted_inputs)
			.for_each(|(error, &input)| {
				// multiply with the derivate to get the delta
				*error *= A::derivate(A::activate(input));
			});
		error
	}

	/// modifies own weights by given deltas
	fn _apply_deltas(&mut self, deltas: GenericArray<f32, Neurons>, inputs: &GenericArray<f32, Input>, speed: f32);

	/// calculates the previous layers errors from this layers errors and weights
	fn _pre_error(&self, deltas: &GenericArray<f32, Neurons>) -> GenericArray<f32, Input>;

	fn teach<F: FnMut(usize, f32), I: IntoIterator<Item = (GenericArray<f32, Input>, GenericArray<f32, FinalOut>)>>(&mut self, lesson : I, iterations: usize, mut callback: F)
		where <I as IntoIterator>::IntoIter: Clone
	{
            let lesson = lesson.into_iter();
            // change modification speed over time
            // start quick, slow down over time
            // todo: make this modifiable by caller
            let factor = |progress| {
                    let percentage = (progress as f32)/(iterations as f32);
                    //let min = 0.08 * percentage;
                    let min = 0.4 * percentage;
                    let max = 0.9 * (1.-percentage);
                    min + max
            };
            for i in 0..iterations {
                let mut avg_loss = 0f32;
                let iter = lesson.clone().enumerate();
                for (pos, (input, output)) in iter {
                            let (_, err) = self.backprop(&input, &output, factor(i));
                            avg_loss = avg_loss + (loss(&err) - avg_loss) / ((pos+1) as f32);
                    }
                callback(i, avg_loss);
            }
	}
}

/// The final layer of a Network.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "Neurons: NL<Input>, Input: AL")]
pub struct OutputLayer<A: Activation, Neurons: NL<Input>, Input: AL> {
	weights: GenericArray<GenericArray<f32, Input>, Neurons>,
	bias: GenericArray<f32, Neurons>,
	phantom: core::marker::PhantomData<A>,
}

impl<A: Activation, Neurons: NL<Input>, Input: AL> OutputLayer<A, Neurons, Input> {
	pub fn new() -> Self {
		let mut out = Self {
			weights: GenericArray::default(),
			bias: GenericArray::default(),
			phantom: core::marker::PhantomData::default(),
		};
		let mut rng = rand::rngs::OsRng;
		for neuron in out.weights.as_mut_slice() {
			for weight in neuron.as_mut_slice() {
				*weight = rng.gen_range(-1., 1.)
			}
		}
		for bias in out.bias.iter_mut() {
			*bias = rng.gen_range(-0.1, 0.1);
		}
		out
	}
}

impl<A: Activation, Input: AL, Neurons: NL<Input>> Layer<Input, Neurons, Neurons, A>
	for OutputLayer<A, Neurons, Input>
{
	fn calculate(&self, inputs: &GenericArray<f32, Input>) -> GenericArray<f32, Neurons> {
		self.step(&self.weight(inputs))
	}
	fn step(&self, inputs: &GenericArray<f32, Neurons>) -> GenericArray<f32, Neurons> {
		inputs
			.into_iter()
			.map(|&input| A::activate(input))
			.collect()
	}

	fn weight(&self, inputs: &GenericArray<f32, Input>) -> GenericArray<f32, Neurons> {
		debug_assert_eq!(self.weights.len(), self.bias.len());
		self.weights
			.iter()
			.zip(&self.bias)
			.map(|(neuron, bias)| {
				neuron
					.iter()
					.zip(inputs.iter())
					.map(|(weight, input)| weight * input)
					.fold(*bias, core::ops::Add::add)
			})
			.collect()
	}

	fn _get_error(
		&mut self,
		mut output: GenericArray<f32, Neurons>,
		correct_output: &GenericArray<f32, Neurons>,
		_speed: f32,
	) -> (GenericArray<f32, Neurons>, GenericArray<f32, Neurons>) {
		// TODO: this might need to be generic
		output
			.iter_mut()
			.zip(correct_output)
			.for_each(|(output, &correct_output)| {
				// subtract the output from the expected output to get the error
				*output = correct_output - *output;
			});
		let error = output;
		(error.clone(), error)
	}

	fn _apply_deltas(
		&mut self,
		mut deltas: GenericArray<f32, Neurons>,
		inputs: &GenericArray<f32, Input>,
		speed: f32,
	) {
		// then add to own weights
		debug_assert_eq!(deltas.len(), self.weights.len());
		debug_assert_eq!(self.bias.len(), self.weights.len());
		deltas.iter_mut().for_each(|d| *d *= speed);
		for ((neuron, delta), bias) in self
			.weights
			.iter_mut()
			.zip(deltas)
			.zip(self.bias.iter_mut())
		{
			debug_assert_eq!(neuron.len(), inputs.len());

			for (weight, &input_activation) in neuron.iter_mut().zip(inputs) {
				*weight += delta * input_activation;
			}
			// bias input activation is always 1
			*bias += delta;
		}
	}

	fn _pre_error(&self, deltas: &GenericArray<f32, Neurons>) -> GenericArray<f32, Input> {
		// first calculate the weighted deltas (basically inverse weighted inputs)
		let mut inverse_delta = GenericArray::<f32, Input>::default();

		for (neuron_weights, neuron_delta) in self.weights.iter().zip(deltas) {
			for (delta, neuron_weight) in inverse_delta.iter_mut().zip(neuron_weights) {
				*delta += *neuron_weight * neuron_delta
			}
		}
		// pass the previous layers errors back up so they may learn from it
		inverse_delta
	}
}

/// a layer that stacks another layer inside itself (which may then recursively stack another and so on)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "Neurons: NL<Input>, Input: AL, NextN: NL<Neurons>, FinalOut: AL")]
pub struct InnerLayer<
	A: Activation,
	Neurons: NL<Input>,
	Input: AL,
	NextN: NL<Neurons>,
	FinalOut: AL,
	NextA: Activation,
	Next: Layer<Neurons, NextN, FinalOut, NextA>,
> {
	inner: OutputLayer<A, Neurons, Input>,
	next: Next,
	phantom: core::marker::PhantomData<(NextN, FinalOut, NextA)>,
}

impl<
		A: Activation,
		Neurons: NL<Input>,
		Input: AL,
		NextN: NL<Neurons>,
		FinalOut: AL,
		NextA: Activation,
		Next: Layer<Neurons, NextN, FinalOut, NextA>,
	> InnerLayer<A, Neurons, Input, NextN, FinalOut, NextA, Next>
{
	/// Pushes this layer on top of an existing layer.
	pub fn push(next: Next) -> Self {
		Self {
			next,
			inner: OutputLayer::new(),
			phantom: core::marker::PhantomData::default(),
		}
	}
}

impl<
		A: Activation,
		Neurons: NL<Input>,
		Input: AL,
		NextN: NL<Neurons>,
		FinalOut: AL,
		NextA: Activation,
		Next: Layer<Neurons, NextN, FinalOut, NextA>,
	> Layer<Input, Neurons, FinalOut, A>
	for InnerLayer<A, Neurons, Input, NextN, FinalOut, NextA, Next>
{
	fn calculate(&self, inputs: &GenericArray<f32, Input>) -> GenericArray<f32, FinalOut> {
		let own_output = self.step(&self.weight(inputs));
		self.next.calculate(&own_output)
	}

	fn weight(&self, inputs: &GenericArray<f32, Input>) -> GenericArray<f32, Neurons> {
		self.inner.weight(inputs)
	}

	fn step(&self, inputs: &GenericArray<f32, Neurons>) -> GenericArray<f32, Neurons> {
		self.inner.step(inputs)
	}

	fn _get_error(
		&mut self,
		output: GenericArray<f32, Neurons>,
		correct_output: &GenericArray<f32, FinalOut>,
		speed: f32,
	) -> (GenericArray<f32, Neurons>, GenericArray<f32, FinalOut>) {
		self.next.backprop(&output, correct_output, speed)
	}

	fn _apply_deltas(
		&mut self,
		deltas: GenericArray<f32, Neurons>,
		inputs: &GenericArray<f32, Input>,
		speed: f32,
	) {
		self.inner._apply_deltas(deltas, inputs, speed)
	}

	fn _pre_error(&self, deltas: &GenericArray<f32, Neurons>) -> GenericArray<f32, Input> {
		self.inner._pre_error(deltas)
	}
}
