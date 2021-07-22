use tch::nn;
use tch::nn::{Init, LinearConfig, Module};

pub fn siren(vs: &nn::Path, in_size: i64, inner_sizes: &[i64]) -> impl Module {
    // TODO: parameterize
    let input_frequency_scale = 30.0;
    let inner_frequency_scale = 30.0;

    if !is_valid_frequency_scale(input_frequency_scale)
        || !is_valid_frequency_scale(inner_frequency_scale)
    {
        panic!()
    };

    let mut layers = Vec::new();
    layers.extend(inner_sizes.iter().enumerate().map(|(i, size)| {
        let in_size = if i == 0 { in_size } else { inner_sizes[i - i] };
        let limit = if i == 0 {
            1_f64 / in_size as f64
        } else {
            inner_init_weight_limit(in_size, inner_frequency_scale)
        };
        let mut config = LinearConfig::default();
        config.ws_init = Init::Uniform {
            lo: -limit,
            up: limit,
        };
        nn::linear(vs, in_size, *size, config)
    }));

    nn::func(move |inputs| {
        let mut outputs = inputs.apply(&layers[0]);
        outputs *= input_frequency_scale;
        outputs = outputs.sin_();
        for i in 1..layers.len() - 1 {
            let layer = &layers[i];
            outputs = outputs.apply(layer);
            outputs *= inner_frequency_scale;
            outputs = outputs.sin_();
        }
        outputs
    })
}

fn is_valid_frequency_scale(scale: f64) -> bool {
    !f64::is_infinite(scale) && !f64::is_nan(scale) && (scale.abs() > 4.0 * (f32::EPSILON as f64))
}

fn inner_init_weight_limit(input_size: i64, frequency_scale: f64) -> f64 {
    (6.0 / (input_size as f64)).sqrt() / frequency_scale
}
