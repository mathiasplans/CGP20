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

pub mod icosphere;
pub mod renderer;
pub mod geometry;
pub mod terraplanet;
pub mod color;
pub mod sun;
pub mod camera;
pub mod initconf;
pub mod idgen;

use icosphere::Icosphere;
use renderer::Renderer;
use terraplanet::TerraPlanet;
use sun::Sun;
use initconf::random_config;


fn main() {
    let r = Renderer::setup();
    let device = r.get_device();

    let c = random_config(device, 12);

    static mut SUNS: Vec<Sun> = Vec::new();
    static mut TERRAPLANETS: Vec<TerraPlanet> = Vec::new();

    unsafe {
        for a in c.0 {
            SUNS.push(a);
        }

        for a in c.1 {
            TERRAPLANETS.push(a);
        }
        r.start(&SUNS, &TERRAPLANETS);
    }
}