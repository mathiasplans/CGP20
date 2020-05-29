
use std::sync::Arc;

use std::collections::HashMap;
use std::cmp;

use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, PersistentDescriptorSetBuilder};
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

use vulkano::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuf;
use vulkano::buffer::cpu_pool::CpuBufferPoolSubbuffer;
use vulkano::memory::pool::StdMemoryPool;


use std::time::Instant;
use cgmath::{Matrix3, Matrix4, Point3, Vector3, Vector4, Rad};

use std::iter;

use crate::geometry::{Vertex, Normal, Point, Vector, Face};
use crate::icosphere::Icosphere;
use crate::color::RGBA;
use crate::camera::Camera;

// Inspired by http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
pub struct Sun {
    device: Arc<Device>,

    vertex_buffer: Option<Arc<CpuAccessibleBuffer<[Vertex]>>>,
    normal_buffer: Option<Arc<CpuAccessibleBuffer<[Normal]>>>,
    index_buffer: Option<Arc<CpuAccessibleBuffer<[u16]>>>,
    uniform_buffer: CpuBufferPool<sun_vs::ty::Data>,

    radius: f32,
    mass: f32,
    id: u32,

    delta: Instant,

    translation: Vector3<f32>,
    velocity: Vector3<f32>,

    vertex_shader: sun_vs::Shader,
    fragment_shader: sun_fs::Shader,

    seed: f32
}

impl Sun {
    pub fn new(device: Arc<Device>, radius: f32, mass: f32, position: Vector3<f32>, id: u32, seed: f32, velocity: Vector3<f32>) -> Self {
        let mut is = Sun {
            device: device.clone(),

            vertex_buffer: None,
            normal_buffer: None,
            index_buffer: None,
            uniform_buffer: CpuBufferPool::<sun_vs::ty::Data>::new(device.clone(), BufferUsage::all()),

            radius: radius,
            mass: mass,
            id: id,

            delta: Instant::now(),

            translation: position,
            velocity: velocity,

            vertex_shader: sun_vs::Shader::load(device.clone()).unwrap(),
            fragment_shader: sun_fs::Shader::load(device.clone()).unwrap(),

            seed: seed
        };

        is.build_buffers(Icosphere::new(radius, 6));

        is
    }

    pub fn build_buffers(&mut self, sphere: Icosphere) {
        let vectors = sphere.get_vectors();

        self.vertex_buffer = Some(CpuAccessibleBuffer::from_iter(self.device.clone(), BufferUsage::vertex_buffer(), false, vectors.0.iter().cloned()).unwrap());
        self.normal_buffer = Some(CpuAccessibleBuffer::from_iter(self.device.clone(), BufferUsage::vertex_buffer(), false, vectors.1.iter().cloned()).unwrap());
        self.index_buffer = Some(CpuAccessibleBuffer::from_iter(self.device.clone(), BufferUsage::index_buffer(), false, vectors.2.iter().cloned()).unwrap());

    }

    pub fn get_buffers(&self) -> (Arc<CpuAccessibleBuffer<[Vertex]>>, Arc<CpuAccessibleBuffer<[Normal]>>, Arc<CpuAccessibleBuffer<[u16]>>) {
        ((*self.vertex_buffer.as_ref().unwrap()).clone(), (*self.normal_buffer.as_ref().unwrap()).clone(), (*self.index_buffer.as_ref().unwrap()).clone())
    }

    pub fn get_pipeline(&self, device: Arc<Device>, render_pass: Arc<dyn RenderPassAbstract + Send + Sync>, dimensions: [u32; 2]) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input(TwoBuffersDefinition::<Vertex, Normal>::new())
            .vertex_shader(self.vertex_shader.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .viewports(iter::once(Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0 .. 1.0,
            }))
            .fragment_shader(self.fragment_shader.main_entry_point(), ())
            .depth_stencil_simple_depth()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .cull_mode_back()
            .polygon_mode_fill() // can be _line for wireframe
            .build(device.clone())
            .unwrap());

        pipeline
    }

    pub fn get_uniforms(&self, camera: &Camera, dimensions: [u32; 2]) -> Arc<CpuBufferPoolSubbuffer<sun_vs::ty::Data, Arc<StdMemoryPool>>> {
        // let elapsed = self.delta.elapsed();

        let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
        let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100000.0);
        let view = Matrix4::look_at(
            camera.get_position(),
            camera.get_lookat(),
            camera.get_up()
        );

        let uniform_data = sun_vs::ty::Data {
            _dummy0: [0, 0, 1, 0, 0, 0, 0, 0],
            _dummy1: [0, 0, 0, 0],
            viewMatrix: view.into(),
            projectionMatrix: proj.into(),
            viewPosition: camera.get_position().into(),

            id: self.id,
            seed: self.seed,
            size: self.radius,
            primaryColor: RGBA::new(0xF1E3D1, 0.0).as_rgb(),
            secondaryColor: RGBA::new(0xFF00FF, 0.0).as_rgb(),
            obliquity: 0.1
        };

        Arc::new(self.uniform_buffer.next(uniform_data).unwrap())
    }

    pub fn get_position(&self) -> [f32; 3] {
        [self.translation.x, self.translation.y, self.translation.z]
    }

    pub fn get_mass(&self) -> f32 {
        self.mass
    }

    pub fn get_rad(&self) -> f32 {
        self.radius
    }

    pub fn get_id(&self) -> u32 {
        self.id
    }

    pub fn get_velocity(&self) -> [f32; 3] {
        [self.velocity.x, self.velocity.y, self.velocity.z]
    }
}

mod sun_vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        include: ["src/shaders"],
        path: "src/shaders/star.vert"
    }
}

mod sun_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        include: ["src/shaders"],
        path: "src/shaders/star.frag"
    }
}

#[allow(dead_code)]
const X: &str = include_str!("shaders/star.vert");
#[allow(dead_code)]
const Y: &str = include_str!("shaders/star.frag");
#[allow(dead_code)]
const Z: &str = include_str!("shaders/lighting.glsl");
#[allow(dead_code)]
const A: &str = include_str!("shaders/noise.glsl");