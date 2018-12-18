use generic_array::typenum;
use generic_array::GenericArray;
use itertools::Itertools;

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use bincode;
use gnuplot;

use dumbnet::activation::Sigmoid;
use dumbnet::layers::{InnerLayer, Layer, OutputLayer};

fn read_gnuplot<R: BufRead>(rdr: &mut R) -> (Vec<f32>, Vec<f32>) {
    rdr.lines()
        .map(|line| {
            let line = line.expect("failed to read file");
            let mut split = line.split(' ');
            let x: f32 = split.next().unwrap().parse().unwrap();
            let y: f32 = split.next().unwrap().parse().unwrap();
            (x, y)
        })
        .unzip()
}
pub fn main() {
    let bottom = OutputLayer::<Sigmoid, typenum::U1, _>::new();
    let hidden_layer = InnerLayer::<Sigmoid, typenum::U4, _, _, _, _, _>::push(bottom);
    let mut input_layer =
        InnerLayer::<Sigmoid, typenum::U4, typenum::U2, _, _, _, _>::push(hidden_layer);

    /*
    let first = arr![u8; 2,3,4,5];
    let second = arr![u8; 6,7,8,9];
    */

    // multiplication works a lot better when the ranges overlap
    let first = vec![2u8, 7, 4, 9];
    let second = vec![6u8, 3, 8, 5];

    // transform integers into -1..1 for the network
    let iter = first
        .into_iter()
        .cartesian_product(second.into_iter())
        .map(|(fst, snd)| {
            let input: GenericArray<f32, _> = [f32::from(fst) / 100., f32::from(snd) / 100.].into();
            let output = f32::from(fst * snd) / 100.;
            (input, [output].into())
        });

    if let Ok(file) = File::open("multiply_network") {
        println!("reading network from disk");
        input_layer = bincode::deserialize_from(&file).unwrap();
    }

    let mut args = std::env::args();
    let _name = args.next();
    let task = args.next().unwrap_or("loss".into());

    match task.as_ref() {
        "reset" => {
            println!("resetting network");
            use std::fs::remove_file;
            remove_file("multiply_data").unwrap();
            remove_file("multiply_network").unwrap();
        }
        "train" => {
            let iterations = args
                .next()
                .map(|s| s.parse().ok())
                .flatten()
                .unwrap_or(2000usize);
            println!("training for {} iterations", iterations);

            let mut data_file = BufWriter::new(File::create("multiply_data").unwrap());

            let progress = indicatif::ProgressBar::new(iterations as u64);
            progress.set_message("training");

            input_layer.teach(iter.clone(), iterations, |iter, loss| {
                if iter % ((iterations / 100) + 1) == 0 {
                    progress.set_position(iter as u64);
                    data_file
                        .write_all(format!("{} {}\n", iter, loss).as_bytes())
                        .unwrap()
                }
            });

            progress.finish();
            let mut net_file = File::create("multiply_network").unwrap();
            bincode::serialize_into(&mut net_file, &input_layer).unwrap();
        }

        _ => {
            println!("displaying loss data");
            let mut data_file =
                BufReader::new(File::open("multiply_data").expect("failed to open file"));
            let (x, y) = read_gnuplot(&mut data_file);
            let mut fg = gnuplot::Figure::new();
            fg.axes2d().lines(&x, &y, &[]);
            fg.show().expect("could not show gnuplot");

            for (input, output) in iter.clone() {
                let result = input_layer.calculate(&input);
                println!(
                    "trained result of {:?} is {} should be {}",
                    input, result[0], output[0]
                );
            }

            println!("swapped inputs");
            for (mut input, output) in iter.clone() {
                input.swap(0, 1);
                let result = input_layer.calculate(&input);
                println!(
                    "trained result of {:?} is {} should be {}",
                    input, result[0], output[0]
                );
            }
        }
    }
}

pub fn read_gnuplot_data<R: std::io::BufRead>(rdr: R) -> (Vec<f32>, Vec<f32>) {
    rdr.lines()
        .map(|line| {
            let line = line.expect("failed to read file");
            let mut split = line.split(' ');
            let x: f32 = split.next().unwrap().parse().unwrap();
            let y: f32 = split.next().unwrap().parse().unwrap();
            (x, y)
        })
        .unzip()
}
