use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

use vulkano::device::{Device, DeviceExtensions};

use cgmath::Vector3;

use crate::sun::Sun;
use crate::terraplanet::TerraPlanet;
use crate::idgen::IDGenerator;
use crate::lavaplanet::LavaPlanet;

pub fn system_config(device: Arc<Device>) -> (Vec<Sun>, Vec<TerraPlanet>, Vec<LavaPlanet>) {
    // ID generator
    let mut id = IDGenerator::new();

    // RNG
    let mut rng = StdRng::seed_from_u64(28941092);

    let suns = vec![Sun::new(device.clone(), 10.0, 500.0, Vector3::new(0.0, 0.0, -15.0), id.get(), rng.gen(),
        Vector3::new(
            0.0,
            0.0,
            0.0
        )
    )];

    let terraplanets = vec![
        TerraPlanet::new(
            device.clone(),
            2.0, 3.0,
            Vector3::new(60.0, 0.0, -15.0),
            id.get(),
            rng.gen(),
            Vector3::new(
                0.0, 0.3, -3.5
            )
        ),

        TerraPlanet::new(
            device.clone(),
            4.0, 7.0,
            Vector3::new(0.0, 0.0, -220.0),
            id.get(),
            rng.gen(),
            Vector3::new(
                -1.6, -0.1, 0.0
            )
        )
    ];

    let lavaplanets = vec![
        LavaPlanet::new(
            device.clone(),
            0.5, 1.0,
            Vector3::new(0.0, 0.0, 10.0),
            id.get(),
            rng.gen(),
            Vector3::new(
                4.0,
                0.0,
                0.0
            )
        )
    ];

    (suns, terraplanets, lavaplanets)
}

pub fn stable_config(device: Arc<Device>) -> (Vec<Sun>, Vec<TerraPlanet>, Vec<LavaPlanet>) {
    // ID generator
    let mut id = IDGenerator::new();

    // RNG
    let mut rng = StdRng::seed_from_u64(28941092);

    let suns = vec![Sun::new(device.clone(), 0.3, 3.0 * 1.0, Vector3::new(3.0 * 0.97000436, 3.0 * -0.24308753, -10.0), id.get(), rng.gen(),
        Vector3::new(
            0.93240737 / 2.0,
            0.86473146 / 2.0,
            0.0
        )
    )];

    let terraplanets = vec![TerraPlanet::new(device.clone(), 0.3, 3.0 * 1.0, Vector3::new(3.0 * -0.97000436, 3.0 * 0.24308753, -10.0), id.get(), rng.gen(),
        Vector3::new(
            0.93240737 / 2.0,
            0.86473146 / 2.0,
            0.0
        )
    )];

    let lavaplanets = vec![LavaPlanet::new(device.clone(), 0.3, 3.0 * 1.0, Vector3::new(0.0, 0.0, -10.0), id.get(), rng.gen(),
        Vector3::new(
            -0.93240737,
            -0.86473146,
            0.0
        )
    )];

    (suns, terraplanets, lavaplanets)
}

pub fn random_config(device: Arc<Device>, nr_of_bodies: u16) -> (Vec<Sun>, Vec<TerraPlanet>, Vec<LavaPlanet>) {
    // ID generator
    let mut id = IDGenerator::new();

    // RNG
    let mut rng = StdRng::seed_from_u64(10100);

    let suns = vec![Sun::new(device.clone(), 2.4, 40.0, Vector3::new(3.0, -1.0, -12.0), id.get(), rng.gen(),
        Vector3::new(
            rng.gen::<f32>(),
            4.0 * rng.gen::<f32>(),
            rng.gen::<f32>()
        )
    )];

    let mut planets = Vec::<TerraPlanet>::new();
    for i in 1..nr_of_bodies/2 {
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
            rng.gen::<f32>(),
            Vector3::new(
                rng.gen::<f32>(),
                rng.gen::<f32>(),
                2.0 * rng.gen::<f32>()
            )
        ));
    }

    let mut lavaplanets = Vec::<LavaPlanet>::new();
    for i in nr_of_bodies/2..nr_of_bodies {
        let size = (rng.gen::<f32>() + 0.2) / 1.5;
        lavaplanets.push(LavaPlanet::new(
            device.clone(),
            size,
            (4.0 + size) * (3.0 + size),
            Vector3::new(
                10.0 * rng.gen::<f32>(),
                15.0 * rng.gen::<f32>(),
                -20.0 * rng.gen::<f32>()
            ),
            id.get(),
            rng.gen::<f32>(),
            Vector3::new(
                3.0 * rng.gen::<f32>(),
                rng.gen::<f32>(),
                rng.gen::<f32>()
            )
        ));
    }

    (suns, planets, lavaplanets)
}