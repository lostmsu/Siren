use std::{convert::TryInto, time::SystemTime, usize};

use siren_torch::{SirenConfig, siren};
use tch::{Reduction, Tensor, nn::{self, LinearConfig, Module, OptimizerConfig}};

#[path = "../common/tensor_tools.rs"]
mod tensor_tools;


extern crate hound;
extern crate siren_torch;
extern crate tch;

fn main() {
    let max_epochs = 10000;
    let passes_per_epoch = 10;
    let save_min_interval = 10;

    let source_path = std::env::args().nth(1).expect("no file given");
    let reader = hound::WavReader::open(&source_path).unwrap();
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let samples: Vec<f32> = reader.into_samples::<f32>().map(|s| s.unwrap()).collect();

    let cuda = tch::Cuda::is_available();
    let device = if cuda {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };

    let vs = nn::VarStore::new(device);

    let width = 512;
    const LAYER_COUNT: usize = 4;

    let mut siren_config = SirenConfig::sound_default();
    // may need to be proprtional to length
    siren_config.input_frequency_scale *= 20.0;

    let siren = nn::seq()
        .add(siren(&vs.root(), 1, &[width; LAYER_COUNT], siren_config))
        .add(nn::linear(&vs.root(), width, 1, LinearConfig::default()))
        .add_fn(|t| t.clamp(-1.0, 1.0));

    let batch_size = if cuda { 64 * 1024 } else { 128 };
    let lr = 3e-6;
    let mut optimizer = nn::Adam::default().build(&vs, lr).unwrap();

    let coords: Vec<f32> = (0..samples.len()).map(|i| i as f32 * 2.0 / samples.len() as f32 - 1.0).collect();
    let feedable_coords = Tensor::of_slice(&coords).to(device).reshape(&[coords.len() as i64, 1]);
    let feedable_samples = Tensor::of_slice(&samples).to(device).reshape(&[samples.len() as i64, 1]);

    let batch_count = samples.len() * passes_per_epoch / batch_size;

    let mut best_loss = f64::INFINITY;
    let mut last_save = 0;

    for epoch in 0..max_epochs {
        let mut total_loss = 0.0;
        let epoch_start_time = SystemTime::now();
        for _batch_n in 0..batch_count {
            let (ins, outs) =
                tensor_tools::random_batch(&feedable_coords, &feedable_samples, batch_size, device);
            optimizer.zero_grad();
            let predicted = siren.forward(&ins);
            let batch_loss = predicted.mse_loss(&outs, Reduction::Mean);
            optimizer.backward_step(&batch_loss);

            let _no_grad = tch::no_grad_guard();
            total_loss += f64::from(
                batch_loss
                    .detach()
                    .to(tch::Device::Cpu)
                    .mean(tch::Kind::Float),
            );
        }

        println!(
            "epoch: {:4} train loss: {:8.5} in {:4}s",
            epoch,
            total_loss / batch_count as f64,
            epoch_start_time.elapsed().unwrap().as_secs()
        );

        if total_loss < best_loss {
            best_loss = total_loss;
            if last_save + save_min_interval < epoch {
                last_save = epoch;
                let save_start_time = SystemTime::now();
                let siren_path = format!("{}.siren", source_path);
                vs.save(siren_path).unwrap();
                render(&siren, &feedable_coords, batch_size, sample_rate, "siren_sample.wav");
                println!(
                    "saved weights and resampled version in {}ms",
                    save_start_time.elapsed().unwrap().as_millis()
                );
            }
        }
    }
}

fn render(siren: &dyn Module, coords: &Tensor, batch_size: usize, sample_rate: u32, path: &str) {
    let _no_grad = tch::no_grad_guard();
    let sample_count = coords.size2().unwrap().0 as i64;
    let samples = batch_forward(siren, &coords, batch_size.try_into().unwrap());
    let samples = samples.reshape(&[sample_count]).to(tch::Device::Cpu);
    let mut sample_buf = vec![0_f32; sample_count as usize];
    samples.copy_data(&mut sample_buf, sample_count as usize);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for sample in sample_buf.iter() {
        writer.write_sample(*sample).unwrap();
    }
}

fn batch_forward(module: &dyn Module, xs: &Tensor, batch_size: i64) -> Tensor {
    let ys_batches: Vec<_> = xs
        .split(batch_size, 0)
        .into_iter()
        .map(|xs_batch| module.forward(&xs_batch))
        .collect();
    Tensor::cat(&ys_batches, 0)
}