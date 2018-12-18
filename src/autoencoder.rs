#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    bound = "InOut: AL, Compressed: AL, EncoderNeurons: NL<InOut>, DecoderNeurons: NL<Compressed>"
)]
pub struct Autoencoder<
    InOut: AL,
    Compressed: AL,
    EncoderNeurons: NL<InOut>,
    DecoderNeurons: NL<Compressed>,
    Encoder: Layer<InOut, EncoderNeurons, Compressed>,
    Decoder: Layer<Compressed, DecoderNeurons, InOut>,
> {
    encoder: Encoder,
    decoder: Decoder,
    phantom: std::marker::PhantomData<(InOut, Compressed, EncoderNeurons, DecoderNeurons)>,
}
impl<
        InOut: AL,
        Compressed: AL,
        EncoderNeurons: NL<InOut>,
        DecoderNeurons: NL<Compressed>,
        Encoder: Layer<InOut, EncoderNeurons, Compressed>,
        Decoder: Layer<Compressed, DecoderNeurons, InOut>,
    > Autoencoder<InOut, Compressed, EncoderNeurons, DecoderNeurons, Encoder, Decoder>
{
    pub fn new(encoder: Encoder, decoder: Decoder) -> Self {
        Self {
            encoder,
            decoder,
            phantom: std::marker::PhantomData::default(),
        }
    }
    pub fn get_encoder(&self) -> &Encoder {
        &self.encoder
    }
    pub fn get_encoder_mut(&mut self) -> &mut Encoder {
        &mut self.encoder
    }
    pub fn get_decoder(&self) -> &Decoder {
        &self.decoder
    }
    pub fn get_decoder_mut(&mut self) -> &mut Decoder {
        &mut self.decoder
    }
    pub fn get_both(self) -> (Encoder, Decoder) {
        (self.encoder, self.decoder)
    }
    fn teach<F: FnMut(usize, f32), I: IntoIterator<Item = GenericArray<f32, InOut>>>(
        &mut self,
        lesson: I,
        iterations: usize,
        mut callback: F,
    ) where
        <I as IntoIterator>::IntoIter: Clone,
    {
        unimplemented!()
    }
}
