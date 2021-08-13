use std::borrow::Borrow;

use tch::nn::{self, Linear, Path};
use tch::nn::{Init, LinearConfig, Module};
use tch::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct SirenConfig {
    pub input_frequency_scale: f64,
    pub inner_frequency_scale: f64,
}

impl Default for SirenConfig {
    fn default() -> Self {
        SirenConfig {
            input_frequency_scale: 30.0,
            inner_frequency_scale: 30.0,
        }
    }
}

impl SirenConfig {
    pub fn sound_default() -> Self {
        SirenConfig {
            input_frequency_scale: 3000.0,
            inner_frequency_scale: 30.0,
        }
    }
}

#[derive(Debug)]
pub struct Siren {
    layers: Vec<Linear>,
    pub config: SirenConfig,
}

pub fn siren<'a, T: Borrow<Path<'a>>>(
    vs: T,
    in_size: i64,
    inner_sizes: &[i64],
    config: SirenConfig,
) -> Siren {
    if !is_valid_frequency_scale(config.input_frequency_scale)
        || !is_valid_frequency_scale(config.inner_frequency_scale)
    {
        panic!()
    };

    let mut layers = Vec::new();
    for i in 0..inner_sizes.len() {
        let in_size = if i == 0 { in_size } else { inner_sizes[i - i] };
        let limit = if i == 0 {
            1.0 / in_size as f64
        } else {
            inner_init_weight_limit(in_size, config.inner_frequency_scale)
        };
        let mut config = LinearConfig::default();
        config.ws_init = Init::Uniform {
            lo: -limit,
            up: limit,
        };
        layers.push(nn::linear(vs.borrow(), in_size, inner_sizes[i], config))
    }

    Siren { layers, config }
}

impl Module for Siren {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut outputs = xs.apply(&self.layers[0]);
        outputs *= self.config.input_frequency_scale;
        outputs = outputs.sin_();
        for i in 1..self.layers.len() - 1 {
            let layer = &self.layers[i];
            outputs = outputs.apply(layer);
            outputs *= self.config.inner_frequency_scale;
            outputs = outputs.sin_();
        }
        outputs
    }
}

impl Siren {
    pub fn truncate(&mut self, layer_count: usize) {
        self.layers.truncate(layer_count);
    }
}

fn is_valid_frequency_scale(scale: f64) -> bool {
    !f64::is_infinite(scale) && !f64::is_nan(scale) && (scale.abs() > 4.0 * (f32::EPSILON as f64))
}

fn inner_init_weight_limit(input_size: i64, frequency_scale: f64) -> f64 {
    (6.0 / (input_size as f64)).sqrt() / frequency_scale
}
