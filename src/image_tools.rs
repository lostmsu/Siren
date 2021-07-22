use std::ops::{Div, Sub};

use image::{ImageBuffer, Pixel};
use tch::{Kind, Tensor};

pub fn normalize_channel_value<T>(v: T) -> T
where
    T: Div<f64, Output = T> + Sub<f64, Output = T>,
{
    v / 128.0 - 1.0
}

pub fn prepare_image<P: 'static>(image: ImageBuffer<P, Vec<u8>>, device: tch::Device) -> Tensor
where
    P: Pixel<Subpixel = u8>,
{
    let height = image.height();
    let width = image.width();
    let channels = P::CHANNEL_COUNT;

    let flattened = image.as_raw();
    let unnormalized = Tensor::of_slice(flattened)
        .reshape(&[(height * width).into(), channels.into()])
        .to(device);
    let normalized = normalize_channel_value(unnormalized.totype(Kind::Float));
    normalized
}

pub fn coord(height: usize, width: usize) -> Tensor {
    let mut result = vec![vec![vec![0.0_f32; 2]; width]; height];
    for y in 0..height {
        for x in 0..width {
            result[y][x][0] = x as f32 * 2.0 / width as f32 - 1.0;
            result[y][x][1] = y as f32 * 2.0 / height as f32 - 1.0;
        }
    }
    let flat: Vec<f32> = result
        .into_iter()
        .flat_map(Vec::into_iter)
        .flatten()
        .collect();
    Tensor::of_slice(&flat).reshape(&[height as i64 * width as i64, 2.into()])
}
