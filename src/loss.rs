use crate::layers::AL;
use core::marker::PhantomData;
use generic_array::GenericArray;

//todo: maybe make loss generic over activations
// and require some extra features
// like sum(prediction) == 1
// or prediction.all > 0
trait Loss<N: AL> {
    fn loss(prediction: &mut GenericArray<f32, N>, reality: &GenericArray<f32, N>);
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Linear<N: AL>(PhantomData<N>);

impl<N: AL> Loss<N> for Linear<N> {
    fn loss(prediction: &mut GenericArray<f32, N>, reality: &GenericArray<f32, N>) {
        prediction.iter_mut().zip(reality).for_each(|(pre, &real)| {
            // subtract the output from the expected output to get the error
            *pre = real - *pre;
        });
    }
}
