// Some code is from: 2016 The vulkano developers
// Licensed under the MIT http://opensource.org/licenses/MIT.

#![allow(unused_imports)]

use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::image::attachment::AttachmentImage;
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace, FullscreenExclusive};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;
use winit::window::{WindowBuilder, Window};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::event::{Event, WindowEvent};

use cgmath::{Matrix3, Matrix4, Point3, Vector3, Rad};

use std::iter;
use std::sync::Arc;
use std::time::Instant;

use std::collections::HashMap;
use std::cmp;
use std::env;
use std::process::exit;

pub mod icosphere;
pub mod renderer;
pub mod geometry;
pub mod terraplanet;
pub mod color;
pub mod sun;
pub mod camera;
pub mod initconf;
pub mod idgen;
pub mod lavaplanet;
pub mod skybox;

use icosphere::Icosphere;
use renderer::Renderer;
use terraplanet::TerraPlanet;
use sun::Sun;
use initconf::{random_config, stable_config, system_config};
use lavaplanet::LavaPlanet;


fn main() {
    // Command line arguments
    let args: Vec<String> = env::args().collect();

    let r = Renderer::setup();
    let device = r.get_device();

    let c: (Vec<Sun>, Vec<TerraPlanet>, Vec<LavaPlanet>);
    match args[1].as_str() {
        "random" => c = random_config(device, 15),
        "stable" => c = stable_config(device),
        "system" => c = system_config(device),
        _ => {println!("Second argument has to be: random, stable, or system"); exit(1)}
    }

    let speed: f32 = args[2].parse().unwrap();

    // let c = random_config(device, 12);
    // // let c = stable_config(device);
    // // let c = system_config(device);

    static mut SUNS: Vec<Sun> = Vec::new();
    static mut TERRAPLANETS: Vec<TerraPlanet> = Vec::new();
    static mut LAVAPLANET: Vec<LavaPlanet> = Vec::new();

    unsafe {
        for a in c.0 {
            SUNS.push(a);
        }

        for a in c.1 {
            TERRAPLANETS.push(a);
        }

        for a in c.2 {
            LAVAPLANET.push(a);
        }

        r.start(&SUNS, &TERRAPLANETS, &LAVAPLANET, speed);
    }
}