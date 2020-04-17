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

#[path = "icosphere.rs"] mod icosphere;
#[path = "renderer.rs"] mod renderer;

use icosphere::Icosphere;
use renderer::Renderer;

fn main() {
    let r = renderer::Renderer::setup();
    let device = r.get_device();

    let clunky = Icosphere::new(device.clone(), 0.5, 0, Vector3::new(0.45, 0.0, -1.0));
    // clunky.set_translation(Vector3::new(0.45, 0.0, 0.0));

    let smooth = Icosphere::new(device.clone(), 0.4, 4, Vector3::new(-0.6, 0.0, -1.0));
    // smooth.set_translation(Vector3::new(-0.7, 0.0, 0.0));

    static mut objects: Vec<Icosphere> = Vec::new();

    unsafe {
        objects.push(clunky);
        objects.push(smooth);
        r.start(&objects);
    }
}