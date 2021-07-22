extern crate image;
extern crate tch;

mod gaussian_noise;
mod image_tools;
mod siren;
mod tensor_tools;

use std::{convert::TryInto, time::SystemTime};

use gaussian_noise::gaussian_noise;
use image_tools::coord;
use siren::siren;

use tch::{
    nn::{self, ModuleT, OptimizerConfig},
    Kind, Reduction, TchError,
};

fn main() -> Result<(), TchError> {
    let max_epochs = 10000;
    let passes_per_epoch = 100;
    let save_min_interval = 100;

    let input_noise = 1_f64 / (128_f64 * 1024_f64);
    let output_noise = 1_f64 / 128_f64;

    let cuda = tch::Cuda::is_available();
    let device = if cuda {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };

    let vs = nn::VarStore::new(device);
    let inner_size = 128;
    let siren_module = siren(&vs.root(), 2, &[inner_size; 4]);

    let siren = nn::seq_t()
        .add(gaussian_noise(input_noise))
        .add(siren_module)
        .add(nn::linear(&vs.root(), inner_size, 3, Default::default()))
        .add_fn(|t| {
            t.clamp(
                image_tools::normalize_channel_value(-0.01),
                image_tools::normalize_channel_value(255.01),
            )
        })
        .add(gaussian_noise(output_noise));

    let batch_size = if cuda { 64 * 1024 } else { 128 };

    let lr = 0.00032 * 64_f64 / 128_f64;
    let mut optimizer = nn::Adam::default().build(&vs, lr)?;

    for pic_path in std::env::args().into_iter().skip(1) {
        let dyn_image = image::open(&pic_path);
        let rgb = dyn_image.unwrap().to_rgb8();
        let width = rgb.width() as usize;
        let height = rgb.height() as usize;
        let image_samples = image_tools::prepare_image(rgb, device);

        let coords = coord(height, width).to(device);

        let batch_count = height * width * passes_per_epoch / batch_size;

        let mut best_loss = f64::INFINITY;
        let mut last_save = 0;

        for epoch in 0..max_epochs {
            let mut total_loss = 0.0;
            let epoch_start_time = SystemTime::now();
            for _batch_n in 0..batch_count {
                let (ins, outs) =
                    tensor_tools::random_batch(&coords, &image_samples, batch_size, device);
                optimizer.zero_grad();
                let predicted = siren.forward_t(&ins, true);
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
                "epoch: {:4} train loss: {:8.5} in {:4}ms",
                epoch,
                total_loss / batch_count as f64,
                epoch_start_time.elapsed().unwrap().as_millis()
            );

            if total_loss < best_loss {
                best_loss = total_loss;
                if last_save + save_min_interval < epoch {
                    last_save = epoch;
                    let save_start_time = SystemTime::now();
                    let siren_path = format!("{}.siren", pic_path);
                    vs.save(siren_path)?;
                    render(&siren, width * 2, height * 2, "sample2X.png", device)?;
                    println!(
                        "saved weights and upscaled version in {}ms",
                        save_start_time.elapsed().unwrap().as_millis()
                    );
                }
            }
        }
    }

    Ok(())
}

fn render(
    siren: &dyn nn::ModuleT,
    width: usize,
    height: usize,
    path: &str,
    device: tch::Device,
) -> Result<(), TchError> {
    let mut image = image::RgbImage::new(width.try_into().unwrap(), height.try_into().unwrap());
    let coords = coord(height, width).to(device);
    let _no_grad = tch::no_grad_guard();
    let learned_image = siren.forward_t(&coords, false);
    let bytes = (learned_image * 128 + 128)
        .clip(0, 255)
        .totype(Kind::Uint8)
        .to(tch::Device::Cpu);
    let image_data = image.as_mut();
    bytes.copy_data_u8(image_data, width * height * 3);
    image.save(path).unwrap();
    Ok(())
}
