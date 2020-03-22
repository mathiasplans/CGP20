
use std::sync::Arc;

use std::collections::HashMap;
use std::cmp;

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

use vulkano::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuf;
use vulkano::buffer::cpu_pool::CpuBufferPoolSubbuffer;
use vulkano::memory::pool::StdMemoryPool;


use std::time::Instant;
use cgmath::{Matrix3, Matrix4, Point3, Vector3, Rad};

use std::iter;

#[path = "geometry.rs"] mod geometry;
use geometry::{Vertex, Normal, Point, Vector, Face};

// Inspired by http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
pub struct Icosphere {
    device: Arc<Device>,
    index: u16,
    vertex_vector: Vec<geometry::Vertex>,
    normal_vector: Vec<geometry::Normal>,
    index_vector: Vec<u16>,
    middle_vertex: HashMap<u32, u16>,

    vertex_buffer: Option<Arc<CpuAccessibleBuffer<[geometry::Vertex]>>>,
    normal_buffer: Option<Arc<CpuAccessibleBuffer<[geometry::Normal]>>>,
    index_buffer: Option<Arc<CpuAccessibleBuffer<[u16]>>>,
    uniform_buffer: CpuBufferPool<vs::ty::Data>,

    radius: f32,
    tessellation: u8,

    delta: Instant,

    translation: Vector3<f32>,

    vertex_shader: vs::Shader,
    fragment_shader: fs::Shader,
}

impl Icosphere {
    pub fn new(device: Arc<Device>, radius: f32, tessellation: u8) -> Self {
        let mut is = Icosphere {
            device: device.clone(),
            index: 0,
            vertex_vector: Vec::new(),
            normal_vector: Vec::new(),
            index_vector: Vec::new(),
            middle_vertex: HashMap::new(),

            vertex_buffer: None,
            normal_buffer: None,
            index_buffer: None,
            uniform_buffer: CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::all()),

            radius: radius,
            tessellation: tessellation,

            delta: Instant::now(),

            translation: Vector3::new(0.0, 0.0, 0.0),

            vertex_shader: vs::Shader::load(device.clone()).unwrap(),
            fragment_shader: fs::Shader::load(device.clone()).unwrap()
        };

        // First, create a icosahedron
        // icosahedron can be constructed from 3 orthogonal rectangles. This is
        // the lenght of the side (golden ratio). The other side is 2 * 1.
        let t = (1.0 + 5.0_f32.sqrt()) / 2.0;

        // Rectangle no.1
        is.add_vertex((-1.0,  t, 0.0));
        is.add_vertex(( 1.0,  t, 0.0));
        is.add_vertex((-1.0, -t, 0.0));
        is.add_vertex(( 1.0, -t, 0.0));

        // Rectangle no.2
        is.add_vertex((0.0, -1.0,  t));
        is.add_vertex((0.0,  1.0,  t));
        is.add_vertex((0.0, -1.0, -t));
        is.add_vertex((0.0,  1.0, -t));

        // Rectangle no.3
        is.add_vertex(( t, 0.0, -1.0));
        is.add_vertex(( t, 0.0,  1.0));
        is.add_vertex((-t, 0.0, -1.0));
        is.add_vertex((-t, 0.0,  1.0));

        // Create a vector for holding the faces
        let mut faces: Vec<geometry::Face> = Vec::new();

        faces.push((0, 11, 5));
        faces.push((0, 5, 1));
        faces.push((0, 1, 7));
        faces.push((0, 7, 10));
        faces.push((0, 10, 11));

        faces.push((1, 5, 9));
        faces.push((5, 11, 4));
        faces.push((11, 10, 2));
        faces.push((10, 7, 6));
        faces.push((7, 1, 8));

        faces.push((3, 9, 4));
        faces.push((3, 4, 2));
        faces.push((3, 2, 6));
        faces.push((3, 6, 8));
        faces.push((3, 8, 9));

        faces.push((4, 9, 5));
        faces.push((2, 4, 11));
        faces.push((6, 2, 10));
        faces.push((8, 6, 7));
        faces.push((9, 8, 1));

        // Tessellate the icosahedron to create icosphere
        for _i in 0..is.tessellation {
            let mut newfaces: Vec<geometry::Face> = Vec::new();

            faces.iter().for_each(|x| {
                // Triangulate the triangle
                let newv1 = is.get_middle_point(x.0, x.1);
                let newv2 = is.get_middle_point(x.1, x.2);
                let newv3 = is.get_middle_point(x.2, x.0);

                newfaces.push((x.0, newv1, newv3));
                newfaces.push((x.1, newv2, newv1));
                newfaces.push((x.2, newv3, newv2));
                newfaces.push((newv1, newv2, newv3));
            });

            faces = newfaces;
        }

        // Create index buffer
        faces.iter().for_each(|x| {
            is.index_vector.push(x.0);
            is.index_vector.push(x.1);
            is.index_vector.push(x.2);
        });

        // Create buffers
        is.build_buffers();

        is
    }

    fn add_vertex(&mut self, point: geometry::Point) -> u16 {
        // normalize
        let rad = self.radius;
        let pointarray = [point.0, point.1, point.2];
        let amplitude = pointarray.iter().fold(0.0, |acc, x| acc + x.powi(2)).sqrt();
        let mut normalized = pointarray.iter().map(|x| x / amplitude);

        // Create a normal
        let normal = (normalized.next().unwrap(), normalized.next().unwrap(), normalized.next().unwrap());
        self.normal_vector.push(geometry::Normal {
            normal: normal
        });

        // Add to vertex_buffer
        self.vertex_vector.push(geometry::Vertex {
            position: (normal.0 * rad, normal.1 * rad, normal.2 * rad)
        });

        self.index += 1;
        self.index - 1
    }

    // This function gets the index of a point between two other points
    fn get_middle_point(&mut self, p1: u16, p2: u16) -> u16 {

        // Check for the middle point existance
        let sindex = cmp::min(p1, p2);
        let lindex = cmp::max(p1, p2);
        let key = (u32::from(sindex) << 16) | u32::from(lindex);

        // It exists
        if self.middle_vertex.contains_key(&key) {
            *self.middle_vertex.get(&key).unwrap()
        }

        // The middle point does not exist, create it
        else {
            let v1 = self.vertex_vector[usize::from(p1)];
            let v2 = self.vertex_vector[usize::from(p2)];
            
            let newindex = self.add_vertex((
                (v1.position.0 + v2.position.0) / 2.0, 
                (v1.position.1 + v2.position.1) / 2.0, 
                (v1.position.2 + v2.position.2) / 2.0
            ));

            // Add it to the cache
            self.middle_vertex.insert(key, newindex);
            
            newindex
        }
    }

    pub fn build_buffers(&mut self) -> (Arc<CpuAccessibleBuffer<[geometry::Vertex]>>, Arc<CpuAccessibleBuffer<[geometry::Normal]>>, Arc<CpuAccessibleBuffer<[u16]>>) {
        self.vertex_buffer = Some(CpuAccessibleBuffer::from_iter(self.device.clone(), BufferUsage::vertex_buffer(), false, self.vertex_vector.iter().cloned()).unwrap());
        self.normal_buffer = Some(CpuAccessibleBuffer::from_iter(self.device.clone(), BufferUsage::vertex_buffer(), false, self.normal_vector.iter().cloned()).unwrap());
        self.index_buffer = Some(CpuAccessibleBuffer::from_iter(self.device.clone(), BufferUsage::index_buffer(), false, self.index_vector.iter().cloned()).unwrap());

        ((*self.vertex_buffer.as_ref().unwrap()).clone(), (*self.normal_buffer.as_ref().unwrap()).clone(), (*self.index_buffer.as_ref().unwrap()).clone())
    }

    pub fn get_buffers(&self) -> (Arc<CpuAccessibleBuffer<[geometry::Vertex]>>, Arc<CpuAccessibleBuffer<[geometry::Normal]>>, Arc<CpuAccessibleBuffer<[u16]>>) {
        ((*self.vertex_buffer.as_ref().unwrap()).clone(), (*self.normal_buffer.as_ref().unwrap()).clone(), (*self.index_buffer.as_ref().unwrap()).clone())
    }

    pub fn get_pipeline(&self, device: Arc<Device>, render_pass: Arc<dyn RenderPassAbstract + Send + Sync>) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input(TwoBuffersDefinition::<geometry::Vertex, geometry::Normal>::new())
            .vertex_shader(self.vertex_shader.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .viewports(iter::once(Viewport {
                origin: [0.0, 0.0],
                dimensions: [1024.0, 768.0],
                depth_range: 0.0 .. 1.0,
            }))
            .fragment_shader(self.fragment_shader.main_entry_point(), ())
            .depth_stencil_simple_depth()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap());

        pipeline
    }

    pub fn get_uniforms(&self, layout: Arc<UnsafeDescriptorSetLayout>) -> Arc<PersistentDescriptorSet<(
        (),
        PersistentDescriptorSetBuf<CpuBufferPoolSubbuffer<
            vs::ty::Data,
            Arc<StdMemoryPool>
        >>
    )>> {
        let uniform_buffer_subbuffer = {
            let elapsed = self.delta.elapsed();
            let rotation = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
            let rotation = Matrix4::from(Matrix3::from_angle_y(Rad(rotation as f32)));

            let translation = Matrix4::from_translation(self.translation);
            
            let aspect_ratio = 1024.0 / 768.0;
            let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
            let view = Matrix4::look_at(Point3::new(0.3, 0.3, 1.0), Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
            let scale = Matrix4::from_scale(1.0);

            let uniform_data = vs::ty::Data {
                world: (translation * rotation * scale).into(),
                // world: (rotation).into(),
                view: view.into(),
                proj: proj.into(),
            };

            self.uniform_buffer.next(uniform_data).unwrap()
        };

        let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
            .add_buffer(uniform_buffer_subbuffer).unwrap()
            .build().unwrap()
        );

        set
    }

    pub fn set_translation(&mut self, tr: Vector3<f32>) {
        self.translation = tr;
    }
}

mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/shaders/terra.vert"
    }
}

mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/shaders/terra.frag"
    }
}