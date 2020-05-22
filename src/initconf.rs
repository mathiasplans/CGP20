use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

use vulkano::device::{Device, DeviceExtensions};

use cgmath::Vector3;

use crate::sun::Sun;
use crate::terraplanet::TerraPlanet;
use crate::idgen::IDGenerator;

pub fn random_config(device: Arc<Device>, nr_of_bodies: u16) -> (Vec<Sun>, Vec<TerraPlanet>) {
    // ID generator
    let mut id = IDGenerator::new();

    // RNG
    let mut rng = StdRng::seed_from_u64(10100);

    let suns = vec![Sun::new(device.clone(), 2.4, 40.0, Vector3::new(3.0, -1.0, -12.0), id.get(), rng.gen())];

    let mut planets = Vec::<TerraPlanet>::new();
    for i in 1..nr_of_bodies {
        let size = rng.gen::<f32>() + 0.2;
        planets.push(TerraPlanet::new(
            device.clone(),
            size,
            (4.0 + size) * (3.0 + size),
            Vector3::new(
                10.0 * rng.gen::<f32>(),
                15.0 * rng.gen::<f32>(),
                -20.0 * rng.gen::<f32>()
            ),
            id.get(),
            rng.gen::<f32>()));
    }

    (suns, planets)
}