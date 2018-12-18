use core::fmt::Debug;
use serde::de::DeserializeOwned;
use serde::Serialize;
pub trait Activation: Debug + Serialize + DeserializeOwned + Clone {
    fn activate(inputs: f32) -> f32;
    fn derivate(activation: f32) -> f32;
}

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
