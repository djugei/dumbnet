use generic_array::typenum;

use dumbnet::prelude::*;

/// ReLu does not really work with inputs outside of [0,1] even though it should in theory at least
/// be able to handle much bigger inputs
fn main() {
	let mut last = OutputLayer::<ReLu, typenum::U1, typenum::U1>::new();
	let iter = (1..5).map(|i| ([i as f32 / 100.].into(), [(2 * i) as f32 / 100.].into()));
	println!("untrained layer: {:?}", last);

	last.teach(iter.clone(), 4000, |_, _| {});

	for (input, output) in iter {
		let result = last.calculate(&input);
		println!(
			"trained result of {:?} is {} should be {}",
			input, result[0], output[0]
		);
	}
	println!("trained layer: {:?}", last);
	let input = [10f32 / 100.].into();
	let result = last.calculate(&input);
	println!(
		"untrained result of {:?} is {} should be {}",
		input,
		result[0],
		input[0] * 2.
	);
}
