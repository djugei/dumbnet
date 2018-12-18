use core::ops::{Add, Div, Mul, Sub};
use generic_array::typenum::bit::B1;
use generic_array::typenum::{Add1, Diff, Prod, Quot, Unsigned};
use generic_array::GenericArray;

use crate::activation::Activation;
use crate::layers::{Layer, AL, NL};

use ndarray::{ArrayView3, ArrayViewMut3};

type ConvOutputSize<W, H, CW, CH, N, S> = Prod<Prod<ConvHSize<H, CH, S>, ConvWSize<W, CW, S>>, N>;
type ConvHSize<H, CH, S> = Add1<Quot<Diff<H, CH>, S>>;
type ConvWSize<W, CW, S> = Add1<Quot<Diff<W, CW>, S>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    bound = "W: AL, H: AL, D: AL, CW: AL, CH: AL, N: AL, S: AL, FinalOut: AL, NextN: NL<ConvOutputSize<W, H, CW, CH, N, S>>, NextA: Activation"
)]
pub struct ConvolutionalLayer<
    A: Activation,
    // width of the input
    W: AL,
    // height of the input
    H: AL,
    // number of inputs/depth of input
    D: AL,
    // width of the filter
    CW: AL,
    // height of the filter
    CH: AL,
    // number of filters
    N: NL<Prod<Prod<CH, CW>, D>>,
    // stride of the filter (how many elements its moved on each application)
    // sadly the currently used matrix library does not allow for having different H and W strides
    S: AL,
    FinalOut: AL,
    NextN: NL<ConvOutputSize<W, H, CW, CH, N, S>>,
    NextA: Activation,
    Next: Layer<ConvOutputSize<W, H, CW, CH, N, S>, NextN, FinalOut, NextA>,
> where
    H: Mul<W>,
    Prod<H, W>: Mul<D>,
    Prod<Prod<H, W>, D>: AL,

    CH: Mul<CW>,
    Prod<CH, CW>: Mul<D>,
    Prod<Prod<CH, CW>, D>: AL,

    H: Sub<CH>,
    Diff<H, CH>: Div<S>,
    Quot<Diff<H, CH>, S>: Add<B1>,

    W: Sub<CW>,
    Diff<W, CW>: Div<S>,
    Quot<Diff<W, CW>, S>: Add<B1>,

    ConvHSize<H, CH, S>: Mul<ConvWSize<W, CW, S>>,
    ConvHSize<H, CH, S>: AL,
    ConvWSize<W, CW, S>: AL,
    Prod<ConvHSize<H, CH, S>, ConvWSize<W, CW, S>>: Mul<N>,

    ConvOutputSize<W, H, CW, CH, N, S>: AL + NL<Prod<Prod<H, W>, D>>,
{
    phantom: core::marker::PhantomData<(A, W, H, D, CW, CH, S, FinalOut, NextN, NextA, Next)>,
    // there are N filters each CH * CW * D in size
    filter: GenericArray<GenericArray<f32, Prod<Prod<CH, CW>, D>>, N>,
    bias: GenericArray<f32, ConvOutputSize<W, H, CW, CH, N, S>>,
    next: Next,
}

impl<
        A: Activation,
        W: AL,
        H: AL,
        D: AL,
        CW: AL,
        CH: AL,
        N: NL<Prod<Prod<CH, CW>, D>>,
        S: AL,
        FinalOut: AL,
        NextN: NL<ConvOutputSize<W, H, CW, CH, N, S>>,
        NextA: Activation,
        Next: Layer<ConvOutputSize<W, H, CW, CH, N, S>, NextN, FinalOut, NextA>,
    > ConvolutionalLayer<A, W, H, D, CW, CH, N, S, FinalOut, NextN, NextA, Next>
where
    H: Mul<W>,
    Prod<H, W>: Mul<D>,
    Prod<H, W>: AL,
    Prod<Prod<H, W>, D>: AL,

    CH: Mul<CW>,
    Prod<CH, CW>: Mul<D>,
    Prod<Prod<CH, CW>, D>: AL,

    H: Sub<CH>,
    Diff<H, CH>: Div<S>,
    Quot<Diff<H, CH>, S>: Add<B1>,

    W: Sub<CW>,
    Diff<W, CW>: Div<S>,
    Quot<Diff<W, CW>, S>: Add<B1>,

    ConvHSize<H, CH, S>: Mul<ConvWSize<W, CW, S>>,
    ConvHSize<H, CH, S>: AL,
    ConvWSize<W, CW, S>: AL,
    Prod<ConvHSize<H, CH, S>, ConvWSize<W, CW, S>>: Mul<N>,

    ConvOutputSize<W, H, CW, CH, N, S>: AL + NL<Prod<Prod<H, W>, D>>,
{
    pub fn push(next: Next) -> Self {
        use rand::Rng;
        let mut rng = rand::rngs::OsRng;

        let mut new = Self {
            next,
            phantom: core::marker::PhantomData::default(),
            filter: GenericArray::default(),
            bias: GenericArray::default(),
        };

        for filter in new.filter.iter_mut() {
            for weight in filter.iter_mut() {
                *weight = rng.gen_range(-1.0f32, 1.0f32);
            }
        }

        for neuron in new.bias.iter_mut() {
            *neuron = rng.gen_range(-0.5, 0.5);
        }

        new
    }
}

impl<
        A: Activation,
        W: AL,
        H: AL,
        D: AL,
        CW: AL,
        CH: AL,
        N: NL<Prod<Prod<CH, CW>, D>>,
        S: AL,
        FinalOut: AL,
        NextN: NL<ConvOutputSize<W, H, CW, CH, N, S>>,
        NextA: Activation,
        Next: Layer<ConvOutputSize<W, H, CW, CH, N, S>, NextN, FinalOut, NextA>,
    > Layer<Prod<Prod<H, W>, D>, ConvOutputSize<W, H, CW, CH, N, S>, FinalOut, A>
    for ConvolutionalLayer<A, W, H, D, CW, CH, N, S, FinalOut, NextN, NextA, Next>
where
    H: Mul<W>,
    Prod<H, W>: Mul<D>,
    Prod<H, W>: AL,
    Prod<Prod<H, W>, D>: AL,

    CH: Mul<CW>,
    Prod<CH, CW>: Mul<D>,
    Prod<Prod<CH, CW>, D>: AL,

    H: Sub<CH>,
    Diff<H, CH>: Div<S>,
    Quot<Diff<H, CH>, S>: Add<B1>,

    W: Sub<CW>,
    Diff<W, CW>: Div<S>,
    Quot<Diff<W, CW>, S>: Add<B1>,

    ConvHSize<H, CH, S>: Mul<ConvWSize<W, CW, S>>,
    ConvHSize<H, CH, S>: AL,
    ConvWSize<W, CW, S>: AL,
    Prod<ConvHSize<H, CH, S>, ConvWSize<W, CW, S>>: Mul<N>,

    ConvOutputSize<W, H, CW, CH, N, S>: AL + NL<Prod<Prod<H, W>, D>>,
{
    fn calculate(
        &self,
        inputs: &GenericArray<f32, Prod<Prod<H, W>, D>>,
    ) -> GenericArray<f32, FinalOut> {
        let own_output = self.step(&self.weight(inputs));
        self.next.calculate(&own_output)
    }

    fn weight(
        &self,
        inputs: &GenericArray<f32, Prod<Prod<H, W>, D>>,
    ) -> GenericArray<f32, ConvOutputSize<W, H, CW, CH, N, S>> {
        let width = <W as Unsigned>::to_usize();
        let height = <H as Unsigned>::to_usize();
        let depth = <D as Unsigned>::to_usize();
        let filter_width = <CW as Unsigned>::to_usize();
        let filter_height = <CH as Unsigned>::to_usize();
        let stride = <S as Unsigned>::to_usize();

        let input = ArrayView3::from_shape((depth, height, width), inputs)
            .expect("input sizes are checked at compile time");

        self.filter
            .iter()
            .flat_map(|arr_filter| {
                let filter =
                    ArrayView3::from_shape((depth, filter_height, filter_width), arr_filter)
                        .expect("filter sizes are checked at compile time");
                // each filter takes the whole depth
                input
                    .windows((depth, filter_height, filter_width))
                    .into_iter()
                    .step_by(stride)
                    .map(move |v| (&v * &filter).sum())
            })
            .zip(&self.bias)
            .map(|(value, bias)| value + bias)
            // filter results are just stored one after another
            // depth is the highest dimension
            .collect()
    }

    fn step(
        &self,
        inputs: &GenericArray<f32, ConvOutputSize<W, H, CW, CH, N, S>>,
    ) -> GenericArray<f32, ConvOutputSize<W, H, CW, CH, N, S>> {
        inputs
            .into_iter()
            .map(|&input| A::activate(input))
            .collect()
    }

    fn _get_error(
        &mut self,
        output: GenericArray<f32, ConvOutputSize<W, H, CW, CH, N, S>>,
        correct_output: &GenericArray<f32, FinalOut>,
        speed: f32,
    ) -> (
        GenericArray<f32, ConvOutputSize<W, H, CW, CH, N, S>>,
        GenericArray<f32, FinalOut>,
    ) {
        self.next.backprop(&output, correct_output, speed)
    }

    fn _apply_deltas(
        &mut self,
        mut deltas: GenericArray<f32, ConvOutputSize<W, H, CW, CH, N, S>>,
        inputs: &GenericArray<f32, Prod<Prod<H, W>, D>>,
        speed: f32,
    ) {
        let width = <W as Unsigned>::to_usize();
        let height = <H as Unsigned>::to_usize();
        let depth = <D as Unsigned>::to_usize();
        let filter_width = <CW as Unsigned>::to_usize();
        let filter_height = <CH as Unsigned>::to_usize();
        let stride = <S as Unsigned>::to_usize();
        let output_height = <ConvHSize<H, CH, S> as Unsigned>::to_usize();
        let output_width = <ConvWSize<W, CW, S> as Unsigned>::to_usize();
        let number_of_filters = <N as Unsigned>::to_usize();

        let input_matrix = ArrayView3::from_shape((depth, height, width), inputs)
            .expect("error sizes are checked at compile time");

        deltas.iter_mut().for_each(|d| *d *= speed);
        self.bias
            .iter_mut()
            .zip(deltas.iter())
            .for_each(|(bias, delta)| *bias += delta);

        let delta_matrix =
            ArrayView3::from_shape((number_of_filters, output_height, output_width), &deltas)
                .expect("error sizes are checked at compile time");

        self.filter
            .iter_mut()
            .zip(delta_matrix.outer_iter())
            .for_each(|(filter, deltas)| {
                let mut filter =
                    ArrayViewMut3::from_shape((depth, filter_height, filter_width), filter)
                        .expect("filter sizes are checked at compile time");
                input_matrix
                    .windows((depth, filter_height, filter_width))
                    .into_iter()
                    .step_by(stride)
                    .zip(deltas.iter())
                    .fold(&mut filter, |filter, (input_activation, &delta)| {
                        filter.scaled_add(delta, &input_activation);
                        filter
                    });
            });
    }

    fn _pre_error(
        &self,
        deltas: &GenericArray<f32, ConvOutputSize<W, H, CW, CH, N, S>>,
    ) -> GenericArray<f32, Prod<Prod<H, W>, D>> {
        let width = <W as Unsigned>::to_usize();
        let height = <H as Unsigned>::to_usize();
        let depth = <D as Unsigned>::to_usize();
        let filter_width = <CW as Unsigned>::to_usize();
        let filter_height = <CH as Unsigned>::to_usize();
        let stride = <S as Unsigned>::to_usize();
        let output_height = <ConvHSize<H, CH, S> as Unsigned>::to_usize();
        let output_width = <ConvWSize<W, CW, S> as Unsigned>::to_usize();
        let number_of_filters = <N as Unsigned>::to_usize();

        let mut pre_errors = GenericArray::default();
        // this is basically a type hint
        if false {
            return pre_errors;
        }
        use core::cell::Cell;
        let errors_cell = Cell::from_mut(pre_errors.as_mut_slice());
        let error_cells = errors_cell.as_slice_of_cells();

        let error_matrix = ArrayView3::from_shape((depth, height, width), error_cells)
            .expect("error sizes are checked at compile time");
        let delta_matrix =
            ArrayView3::from_shape((number_of_filters, output_height, output_width), deltas)
                .expect("error sizes are checked at compile time");

        // sidenote: i spent literally hours looking up smart matrix maths to do this,
        // but i did not find anything usable
        // sidenote2: tensorflow code is fucking unreadable
        // sidenote3: this is crazy slow and very likely buggy
        //
        // the deltas have one 2d matrix per filter
        // first zip the matching ones together
        self.filter
            .iter()
            .zip(delta_matrix.outer_iter())
            .for_each(|(filter, delta)| {
                // apply the filter to the error_matrix for each delta-field
                error_matrix
                    .windows((depth, filter_height, filter_width))
                    .into_iter()
                    .step_by(stride)
                    .zip(delta.iter())
                    .for_each(|(e, &d)| {
                        // matrix lib does not want to play nice with Cell
                        // so we have to manually do the loops
                        e.iter().zip(filter.iter()).for_each(|(e, f)| {
                            e.set(e.get() + (d * f));
                        });
                    });
            });

        pre_errors
    }
}
