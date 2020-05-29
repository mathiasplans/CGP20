
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

pub struct SkyBox {
    device: Arc<Device>,

    vertex_buffer: Option<Arc<CpuAccessibleBuffer<[Vertex]>>>,
    normal_buffer: Option<Arc<CpuAccessibleBuffer<[Normal]>>>,
    index_buffer: Option<Arc<CpuAccessibleBuffer<[u16]>>>,
    uniform_buffer: CpuBufferPool<skybox_vs::ty::Data>,

    radius: f32,

    vertex_shader: skybox_vs::Shader,
    fragment_shader: skybox_fs::Shader,

    seed: f32
}

impl SkyBox {
    pub fn new(device: Arc<Device>) -> Self {
        let radius = 5.0;

        let mut is = SkyBox {
            device: device.clone(),

            vertex_buffer: None,
            normal_buffer: None,
            index_buffer: None,
            uniform_buffer: CpuBufferPool::<skybox_vs::ty::Data>::new(device.clone(), BufferUsage::all()),

            radius: radius,

            vertex_shader: skybox_vs::Shader::load(device.clone()).unwrap(),
            fragment_shader: skybox_fs::Shader::load(device.clone()).unwrap(),

            seed: 0.12
        };

        is.build_buffers(Icosphere::new(radius, 2));

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
                depth_range: 0.99 .. 1.0,
            }))
            .fragment_shader(self.fragment_shader.main_entry_point(), ())
            .depth_stencil_simple_depth()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .cull_mode_front()
            .polygon_mode_fill() // can be _line for wireframe
            .build(device.clone())
            .unwrap());

        pipeline
    }

    pub fn get_uniforms(&self, camera: &Camera, dimensions: [u32; 2]) -> Arc<CpuBufferPoolSubbuffer<skybox_vs::ty::Data, Arc<StdMemoryPool>>> {
        // let elapsed = self.delta.elapsed();

        let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
        let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100000.0);
        let view = Matrix4::look_at(
            camera.get_position(),
            camera.get_lookat(),
            camera.get_up()
        );

        let translation = Matrix4::from_translation(Vector3::new(camera.get_position().x, camera.get_position().y, camera.get_position().z));


        let uniform_data = skybox_vs::ty::Data {
            modelMatrix: translation.into(),
            viewMatrix: view.into(),
            projMatrix: proj.into(),
            seed: self.seed
        };

        Arc::new(self.uniform_buffer.next(uniform_data).unwrap())
    }
}

mod skybox_vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        include: ["src/shaders"],
        path: "src/shaders/skybox.vert"
    }
}

mod skybox_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        include: ["src/shaders"],
        path: "src/shaders/skybox.frag"
    }
}

#[allow(dead_code)]
const X: &str = include_str!("shaders/skybox.vert");
#[allow(dead_code)]
const Y: &str = include_str!("shaders/skybox.frag");
#[allow(dead_code)]
const A: &str = include_str!("shaders/noise.glsl");