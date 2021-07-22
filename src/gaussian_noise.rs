use tch::nn;
use tch::nn::Module;

pub fn gaussian_noise(std_dev: f64) -> impl Module {
    // TODO: check inputs
    nn::func(move |inputs| {
        let mut noise = inputs.randn_like();
        noise *= std_dev;
        inputs + noise
    })
}
