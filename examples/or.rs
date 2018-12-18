use dumbnet::activation::Sigmoid;
use dumbnet::layers::{Layer, OutputLayer};

use generic_array::typenum;

fn main() {
    let mut last = OutputLayer::<Sigmoid, typenum::U1, typenum::U2>::new();
    let inputs = vec![
        ([0., 0.].into(), [0.].into()),
        ([0., 1.].into(), [1.].into()),
        ([1., 0.].into(), [1.].into()),
        ([1., 1.].into(), [1.].into()),
    ];
    last.teach(inputs.clone().into_iter(), 1000, |_, _| {});

    for (input, output) in &inputs {
        let result = last.calculate(&input);
        println!(
            "trained result of {:?} is {} should be {}",
            input, result[0], output[0]
        );
    }
}
