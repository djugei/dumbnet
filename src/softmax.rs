use crate::activation::SoftMax as SA;
use crate::layers::{Layer, AL, NL};
use generic_array::GenericArray;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "Neurons: NL<Input>, Input: AL")]
pub struct SoftMax<Neurons: NL<Input>, Input: AL> {
    weights: GenericArray<GenericArray<f32, Input>, Neurons>,
    bias: GenericArray<f32, Neurons>,
}

impl<Neurons: NL<Input>, Input: AL> SoftMax<Neurons, Input> {
    pub fn new() -> Self {
        let mut out = Self {
            weights: GenericArray::default(),
            bias: GenericArray::default(),
        };
        use rand::Rng;
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

impl<Input: AL, Neurons: NL<Input>> Layer<Input, Neurons, Neurons, SA> for SoftMax<Neurons, Input> {
    fn calculate(&self, inputs: &GenericArray<f32, Input>) -> GenericArray<f32, Neurons> {
        self.step(&self.weight(inputs))
    }

    fn step(&self, inputs: &GenericArray<f32, Neurons>) -> GenericArray<f32, Neurons> {
        // for numerical stability we reduce stuff by the maximum input
        let max = inputs
            .iter()
            .cloned()
            .fold(core::f32::NEG_INFINITY, f32::max);
        let exp: GenericArray<f32, Neurons> = inputs.iter().map(|f| (f - max).exp()).collect();

        let exp_sum = exp.iter().sum::<f32>();
        exp.iter().map(|i| *i / exp_sum).collect()
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

    fn backprop(
        &mut self,
        input: &GenericArray<f32, Input>,
        correct_output: &GenericArray<f32, Neurons>,
        speed: f32,
    ) -> (GenericArray<f32, Input>, GenericArray<f32, Neurons>) {
        let weighted_inputs = self.weight(input);
        let mut output = self.step(&weighted_inputs);

        // online documentation really wants to train multiple samples at the same time
        // i am currently only ever training one sample at a time
        // that might be bad for results...

        // reduce only the correct outputs output by one
        // and invert so its the gradient instead of the direction
        output
            .iter_mut()
            .zip(correct_output)
            .for_each(|(o, c)| *o = -(*o - c));
        let delta = output;

        let pre_error = self._pre_error(&delta);
        self._apply_deltas(delta.clone(), input, speed);

        // the errors are just passed up for informational purposes, so a training alg can
        // determine how wrong the network is without running an extra recognition step
        // its not actually used for backprop at all.
        (pre_error, delta)
    }

    /// weights errors relative to activation. gets called by backprop, don't call this manually
    fn _weight_errors(
        &self,
        _error: GenericArray<f32, Neurons>,
        _weighted_inputs: &GenericArray<f32, Neurons>,
    ) -> GenericArray<f32, Neurons> {
        panic!()
    }

    fn _get_error(
        &mut self,
        _output: GenericArray<f32, Neurons>,
        _correct_output: &GenericArray<f32, Neurons>,
        _speed: f32,
    ) -> (GenericArray<f32, Neurons>, GenericArray<f32, Neurons>) {
        panic!()
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
