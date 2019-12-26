//! Activations for use in [layers](crate::layers)
//!
//! The Enums in this module are empty and therefore not initializable. They are only used as type
//! paramenters.
use core::fmt::Debug;
use serde::de::DeserializeOwned;
use serde::Serialize;

pub trait Activation: Debug + Serialize + DeserializeOwned + Clone {
    /// Input is the summed and weighted inputs of this neuron
    ///
    /// Output is the output of the neuron
    fn activate(inputs: f32) -> f32;

    /// Input is the previous activation, as in the result of this traits activate() function
    ///
    /// Output is the derivate of this activation function at that point.
    /// The derivate may be slightly adapted to better suit the needs of gradient descent
    fn derivate(activation: f32) -> f32;
}

/// A smooth sigmoid between -1 and 1.
///
/// 1/ 1 + (e^input)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Sigmoid {}

impl Activation for Sigmoid {
    #[inline(always)]
    fn activate(input: f32) -> f32 {
        // clamp to make the .exp() in sigmoid not go crazy
        let clamped = input.max(-20.).min(20.);
        let res = 1. / (1. + (-clamped).exp());
        debug_assert!(!res.is_nan());
        res
    }
    #[inline(always)]
    fn derivate(activation: f32) -> f32 {
        debug_assert!(!activation.is_nan());
        let res = activation * (1. - activation);
        debug_assert!(!res.is_nan());
        res
    }
}

/// Returns max(0, input)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ReLu {}

impl Activation for ReLu {
    #[inline(always)]
    fn activate(input: f32) -> f32 {
        input.max(0.)
    }
    #[inline(always)]
    fn derivate(activation: f32) -> f32 {
        if activation > 0. {
            1.
        } else {
            0.01
        }
    }
}

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum SoftMax {}

impl Activation for SoftMax {
    #[inline(always)]
    fn activate(_input: f32) -> f32 {
        panic!()
    }
    #[inline(always)]
    fn derivate(_activation: f32) -> f32 {
        panic!()
    }
}
