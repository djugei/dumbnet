#![no_std]
#![feature(trait_alias)]
#![allow(clippy::type_complexity)]

#[macro_use]
extern crate serde_derive;

extern crate generic_array;

pub mod activation;
//pub mod convolution;
pub mod layers;
pub mod loss;
pub mod prelude;
pub mod softmax;
